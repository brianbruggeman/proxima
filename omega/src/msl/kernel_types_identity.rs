use super::*;

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
    /// Z-extent of the dispatch grid -- `1` for every kernel today. Exists so
    /// [`crate::metal::dispatch`] can grow a third grid axis for a future
    /// batched dispatch (multiple independent same-shape ops sharing one
    /// pipeline, addressed by `threadgroup_position_in_grid.z`) without a
    /// signature change; `1` reproduces today's `MTLSize { depth: 1, .. }`
    /// exactly, so this field is inert until a caller sets it above `1`.
    pub depth: u64,
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
/// let activated = append(
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
/// // `bind` binds only what `outputs` reaches (`bind_plain`'s reachability
/// // pass, ROW 541, `proxima-tensor/docs/discipline.md`) -- an empty
/// // outputs list binds nothing.
/// let bound_ops = proxima_tensor::bind(
///     &program,
///     &shapes,
///     &[activated],
///     proxima_tensor::NumericPolicy::default(),
/// )?;
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

/// `Q5_1`: a flat 32-element block, one `f16` scale AND one `f16` min
/// (unlike `Q4_0`'s single scale) plus a 5th bit per element split out into
/// a separate `qh` plane -- same KIND-difference from the K-quant family
/// [`Q8_0_UNPACK_MSL`]/[`Q4_0_UNPACK_MSL`] draw: no super-block, so this
/// codec does not take the row-blocked (`classify_packed_row_block`) or
/// tiled-GEMM (`classify_tiled_gemm`) fast paths either -- it always
/// renders through the fully generic per-element accessor below.
///
/// Ports `proxima_gguf::quant::q5_1::dequantize_block` exactly (see that
/// function's own doc for the exact `ggml-quants.c` bit-shift derivation):
/// each packed nibble byte carries two 4-bit levels (low nibble at element
/// `j`, high nibble at element `16 + j`), each widened to 5 bits by a
/// per-element bit from `qh`, then `x = level * d + m` -- unlike `Q4_0`, no
/// fixed-midpoint recenter, since the format carries its own `m` term.
///
/// Layout, 24 bytes per 32 elements: `d` f16 at 0, `m` f16 at 2, 4 bytes of
/// packed 5th bits (`qh`) at 4, 16 packed-nibble bytes (`qs`) at 8.
pub const Q5_1_UNPACK_MSL: &str = r#"
// element `index` (0..32) of one Q5_1 block, byte-for-byte the value
// proxima_gguf::quant::q5_1::dequantize_block writes at the same index.
static inline float q5_1_element(device const uchar *block, uint index) {
    ushort d_bits = (ushort)((uint)block[0] | ((uint)block[1] << 8));
    ushort m_bits = (ushort)((uint)block[2] | ((uint)block[3] << 8));
    float delta = (float)as_type<half>(d_bits);
    float minimum = (float)as_type<half>(m_bits);
    uint qh = (uint)block[4] | ((uint)block[5] << 8) | ((uint)block[6] << 16) | ((uint)block[7] << 24);
    bool high_half = index >= 16u;
    uint local = high_half ? (index - 16u) : index;
    uchar byte = block[8u + local];
    uint level;
    if (high_half) {
        uint high_bit = (qh >> (local + 12u)) & 0x10u;
        level = (uint)(byte >> 4u) | high_bit;
    } else {
        uint high_bit = ((qh >> local) << 4u) & 0x10u;
        level = (uint)(byte & 0x0Fu) | high_bit;
    }
    return (float)level * delta + minimum;
}
"#;

/// Bytes one `Q5_1` block occupies -- read from
/// `proxima_gguf::quant::q5_1::BLOCK_BYTES`; pinned in
/// `omega/tests/q5_1_unpack.rs`, same posture as [`Q4_0_BLOCK_BYTES`].
pub const Q5_1_BLOCK_BYTES: usize = proxima_gguf::quant::q5_1::BLOCK_BYTES;

/// Elements one `Q5_1` block carries -- 32, the same flat block width as
/// [`Q4_0_BLOCK_ELEMENTS`], NOT [`Q4K_BLOCK_ELEMENTS`]'s 256. Read from
/// `proxima_gguf::quant::q5_1::QK5_1`.
pub const Q5_1_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q5_1::QK5_1;

/// `Q5_0`: [`Q5_1_UNPACK_MSL`] with the `m` (min) term dropped -- a flat
/// 32-element block, one `f16` scale and a 5th bit per element split out
/// into a separate `qh` plane, same as `Q5_1`, but no per-block min: `x = d
/// * (level - 16)`, a fixed midpoint recenter exactly like [`Q4_0_UNPACK_MSL`]
/// except `level` is 5 bits wide instead of 4. Same KIND-difference from the
/// K-quant family [`Q8_0_UNPACK_MSL`]/[`Q4_0_UNPACK_MSL`] draw: no
/// super-block, so this codec does not take the row-blocked
/// (`classify_packed_row_block`) or tiled-GEMM (`classify_tiled_gemm`) fast
///
/// paths either -- it always renders through the fully generic per-element
/// accessor below.
///
/// Ports `proxima_gguf::quant::q5_0::dequantize_block` exactly (see that
/// function's own doc for the exact `ggml-quants.c` bit-shift derivation):
/// each packed nibble byte carries two 4-bit levels (low nibble at element
/// `j`, high nibble at element `16 + j`), each widened to 5 bits by a
/// per-element bit from `qh`.
///
/// Layout, 22 bytes per 32 elements: `d` f16 at 0, 4 bytes of packed 5th
/// bits (`qh`) at 2, 16 packed-nibble bytes (`qs`) at 6 -- [`Q5_1_UNPACK_MSL`]'s
/// layout with the `m` field at offset 2 removed and every later field
/// shifted back by 2 bytes.
pub const Q5_0_UNPACK_MSL: &str = r#"
// element `index` (0..32) of one Q5_0 block, byte-for-byte the value
// proxima_gguf::quant::q5_0::dequantize_block writes at the same index.
static inline float q5_0_element(device const uchar *block, uint index) {
    ushort d_bits = (ushort)((uint)block[0] | ((uint)block[1] << 8));
    float delta = (float)as_type<half>(d_bits);
    uint qh = (uint)block[2] | ((uint)block[3] << 8) | ((uint)block[4] << 16) | ((uint)block[5] << 24);
    bool high_half = index >= 16u;
    uint local = high_half ? (index - 16u) : index;
    uchar byte = block[6u + local];
    uint level;
    if (high_half) {
        uint high_bit = (qh >> (local + 12u)) & 0x10u;
        level = (uint)(byte >> 4u) | high_bit;
    } else {
        uint high_bit = ((qh >> local) << 4u) & 0x10u;
        level = (uint)(byte & 0x0Fu) | high_bit;
    }
    return ((float)level - 16.0f) * delta;
}
"#;

/// Bytes one `Q5_0` block occupies -- read from
/// `proxima_gguf::quant::q5_0::BLOCK_BYTES`; pinned in
/// `omega/tests/q5_0_unpack.rs`, same posture as [`Q4_0_BLOCK_BYTES`].
pub const Q5_0_BLOCK_BYTES: usize = proxima_gguf::quant::q5_0::BLOCK_BYTES;

/// Elements one `Q5_0` block carries -- 32, the same flat block width as
/// [`Q4_0_BLOCK_ELEMENTS`]/[`Q5_1_BLOCK_ELEMENTS`]. Read from
/// `proxima_gguf::quant::q5_0::QK5_0`.
pub const Q5_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q5_0::QK5_0;

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
    /// Flat 32-element block, one `f16` scale AND one `f16` min, plus a
    /// separate 5th-bit plane -- same KIND-difference from the K-quant
    /// family as [`Self::Q8_0`]/[`Self::Q4_0`]; see [`Q5_1_UNPACK_MSL`]'s
    /// own doc.
    Q5_1,
    /// Flat 32-element block, one `f16` scale, plus a separate 5th-bit
    /// plane -- [`Self::Q5_1`] with the min term dropped, same KIND-
    /// difference from the K-quant family as [`Self::Q8_0`]/[`Self::Q4_0`];
    /// see [`Q5_0_UNPACK_MSL`]'s own doc.
    Q5_0,
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
    /// The codec a raw [`QuantizedBlock`] carries, if any — the one place
    /// every driver (`cuda_driver::packed_codec`, `wgpu_driver::
    /// packed_operands_of`, `metal::device_buffers_arena_plan::
    /// packed_operands_of`) asks "is this operand packed, and under which
    /// codec." Each driver still decides its OWN supported subset (not
    /// every backend has an unpack kernel for every codec this returns
    /// `Some` for) — this only answers whether `PackedCodec` has a variant
    /// for the block at all. `None` for the two non-quantized carriers
    /// ([`QuantizedBlock::Float32`]/`Int32`) and for the codecs with no
    /// `PackedCodec`/unpack-kernel entry anywhere yet
    /// ([`QuantizedBlock::Iq4Nl`]/`Iq2Xs`/`Iq3Xxs`) — decode-only, CPU-side
    /// so far (see `proxima_tensor::cpu`).
    #[cfg(feature = "std")]
    pub(crate) const fn from_quantized_block(block: &QuantizedBlock<'_>) -> Option<Self> {
        match block {
            QuantizedBlock::Q2K(_) => Some(Self::Q2K),
            QuantizedBlock::Q3K(_) => Some(Self::Q3K),
            QuantizedBlock::Q4K(_) => Some(Self::Q4K),
            QuantizedBlock::Q5K(_) => Some(Self::Q5K),
            QuantizedBlock::Q6K(_) => Some(Self::Q6K),
            QuantizedBlock::Q8_0(_) => Some(Self::Q8_0),
            QuantizedBlock::Q4_0(_) => Some(Self::Q4_0),
            QuantizedBlock::Q5_1(_) => Some(Self::Q5_1),
            QuantizedBlock::Q5_0(_) => Some(Self::Q5_0),
            QuantizedBlock::Float16(_) => Some(Self::Float16),
            QuantizedBlock::BFloat16(_) => Some(Self::BFloat16),
            QuantizedBlock::Float32(_)
            | QuantizedBlock::Int32(_)
            | QuantizedBlock::Iq4Nl(_)
            | QuantizedBlock::Iq2Xs(_)
            | QuantizedBlock::Iq3Xxs(_) => None,
        }
    }

    pub(crate) const fn cache_token(self) -> &'static str {
        match self {
            PackedCodec::Q2K => "q2k",
            PackedCodec::Q3K => "q3k",
            PackedCodec::Q4K => "q4k",
            PackedCodec::Q5K => "q5k",
            PackedCodec::Q6K => "q6k",
            PackedCodec::Q8_0 => "q8_0",
            PackedCodec::Q4_0 => "q4_0",
            PackedCodec::Q5_1 => "q5_1",
            PackedCodec::Q5_0 => "q5_0",
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
            PackedCodec::Q5_1 => Q5_1_BLOCK_BYTES,
            PackedCodec::Q5_0 => Q5_0_BLOCK_BYTES,
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
            PackedCodec::Q5_1 => Q5_1_BLOCK_ELEMENTS,
            PackedCodec::Q5_0 => Q5_0_BLOCK_ELEMENTS,
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

