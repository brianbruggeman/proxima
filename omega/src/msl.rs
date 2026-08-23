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
//! reduction is present at all and which [`Keep`] it is. Two `BoundOp`
//! nodes that agree on structure but differ in concrete extents, strides, or
//! which buffers they bind therefore emit byte-identical source — see
//! `same_structure_different_extents_yield_identical_source` below for the
//! proof. This is what makes a kernel cacheable (and an `MTLLibrary`
//! reusable) by structure rather than by node identity.
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
//! from [`type_token`] rather than hardcoding `float`, so a `Float16` node
//! emits a kernel of `half` declarations while a `Float32` node emits the
//! same `float` kernel this module always has. The *op logic* — which
//! `ScalarOp` token, which reduction init, how a body's steps chain — never
//! consults dtype at all: [`op_token`], [`scalar_op_expr`], [`init_token`],
//! [`fold_init_tokens`] stay total over their enums exactly as before, and
//! only the declaration spelling varies. `cpu.rs`'s own evaluator remains
//! f32-only (`cpu::reject_non_float32`) — it is the reference oracle, not
//! this crate's dtype ceiling. `omega::execute` runs its own, narrower
//! upstream gate (`Float32` or `Float16` only) before a `BoundOp` ever
//! reaches [`emit`].

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::collections::BTreeMap;

use proxima_tensor::{
    BoundOp, BoundOpKind, ComposedBody, DType, Keep, Layout, NodeId, ReduceInit, ScalarOp, StepArg,
};

use crate::error::EmitError;
use crate::sized::SIMD_WIDTH;

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
}

/// How many threads a driver must dispatch for this op — one per
/// independent unit of work (output element for elementwise/reduce, output
/// line for a scan). Unlike [`Kernel::source`], this genuinely is a function
/// of the op's concrete extents, not just its structure: it is per-dispatch
/// data, the same way an argument to a function call is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSpec {
    pub threads: u64,
    /// `Some(SIMD_WIDTH)` for a cooperative reduce (see [`reduce_is_cooperative`]):
    /// the driver must dispatch threadgroups exactly this wide so every
    /// SIMD-group boundary lands on an output-element boundary (`gid / SIMD_WIDTH`
    /// is only a valid output index under that alignment — see
    /// `push_cooperative_reduce_body`'s doc). `None` for every other kernel,
    /// which keeps the occupancy-driven width the driver already picks.
    pub threadgroup_width: Option<u64>,
}

/// Emits an MSL kernel from a bound [`BoundOp`] — the GPU-emission half of
/// the same descriptor [`proxima_tensor::cpu`] interprets on CPU. See the
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
/// let bound_ops = proxima_tensor::bind(&program, &shapes, &[])?;
///
/// let kernel = omega::emit(&bound_ops[0])?;
/// assert!(kernel.source.contains("kernel void"));
/// assert!(kernel.source.contains("tanh("));
/// assert_eq!(kernel.bindings.len(), 3); // one input, one output, uniforms
/// assert_eq!(kernel.grid.threads, 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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
/// Ports [`proxima_gguf::quant::q4_k::dequantize_block`] exactly, including
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

// Eight consecutive levels from TWO 32-bit loads instead of eight byte
// loads. A lane's run is `slot .. slot+7` and never crosses a 32-element
// sub-block boundary, so all eight share a group and a nibble half, and
// their bytes are eight CONSECUTIVE bytes of `qs`. `slot % 32` is one of
// {0,8,16,24} and a super-block is 144 bytes (a multiple of 16), so the
// address is 4-byte aligned and the `uint` cast is sound.
//
// ggml does the same thing one width down (`q1[i] & 0x000F / 0x0F00 /
// 0x00F0 / 0xF000` off a `uint16_t`), for the same reason: the nibble
// extract is cheap and the LOAD is what costs.
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
"#;

/// Bytes one `Q4_K` super-block occupies, and elements it carries — the two
/// numbers a caller needs to index a packed weight row. Mirrors
/// `proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K}`; omega does not depend on
/// `proxima-gguf` at build time, so they are restated here and pinned by a
/// test that does.
pub const Q4K_BLOCK_BYTES: usize = 144;
/// Elements one `Q4_K` super-block carries. Shared by `Q5_K`/`Q6_K` too —
/// the whole K-quant super-block family is 256 elements wide
/// (`proxima-tensor/src/cpu.rs`'s own doc on its `Q6K_BLOCK_BYTES` makes the
/// same point); only the packed BYTE width differs per codec.
pub const Q4K_BLOCK_ELEMENTS: usize = 256;

/// MSL source for unpacking one element of a `Q6_K` super-block. Ports
/// [`proxima_gguf::quant::q6_k::dequantize_block`]/`unpack_levels` exactly:
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

/// Bytes one `Q6_K` super-block occupies. Mirrors
/// `proxima_gguf::quant::q6_k::BLOCK_BYTES`; pinned in
/// `omega/tests/q6k_unpack.rs`, same posture as [`Q4K_BLOCK_BYTES`].
pub const Q6K_BLOCK_BYTES: usize = 210;

/// MSL source for unpacking one element of a `Q5_K` super-block. Ports
/// [`proxima_gguf::quant::q5_k::dequantize_block`]/`get_scale_min_k4`
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

/// Bytes one `Q5_K` super-block occupies. Mirrors
/// `proxima_gguf::quant::q5_k::BLOCK_BYTES`; pinned in
/// `omega/tests/q5k_unpack.rs`, same posture as [`Q4K_BLOCK_BYTES`].
pub const Q5K_BLOCK_BYTES: usize = 176;

/// Which packed K-quant codec one operand's bytes are — the second axis
/// [`emit`] needs alongside "is this operand packed at all" (a plain `bool`
/// cannot distinguish `Q4_K`'s 144-byte super-block from `Q6_K`'s 210-byte
/// one, or which unpack function reads it). `Copy`/`Eq` so it can sit
/// directly in the `quantized` slice every render function already threads
/// through, with no allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedCodec {
    Q4K,
    Q5K,
    Q6K,
}

impl PackedCodec {
    /// Bytes one super-block of this codec occupies — the multiplier
    /// [`operand_read`] and the row-blocked path need to step between
    /// super-blocks; element count per super-block is shared
    /// ([`Q4K_BLOCK_ELEMENTS`]) across the whole K-quant family.
    const fn block_bytes(self) -> usize {
        match self {
            PackedCodec::Q4K => Q4K_BLOCK_BYTES,
            PackedCodec::Q5K => Q5K_BLOCK_BYTES,
            PackedCodec::Q6K => Q6K_BLOCK_BYTES,
        }
    }
}

/// Every packed operand a bound program has, keyed by [`NodeId`] to its
/// codec — the single source of truth [`emit`] (via the `quantized` slice it
/// derives) and the Metal driver's `correct_packed_matmul_layouts` call both
/// need, generalizing the Q4_K-only `BTreeSet<NodeId>` this crate carried
/// before Q6_K support existed.
pub type PackedOperands = BTreeMap<NodeId, PackedCodec>;

/// One codec slot per operand: which of `resolved`'s operands is a packed
/// buffer (and which [`PackedCodec`]) rather than a flat element array.
/// Shared by [`emit`] and the cheap pre-compile helpers below so the three
/// never re-derive it differently.
fn operand_codecs(resolved: &BoundOp, packed_operands: &PackedOperands) -> Vec<Option<PackedCodec>> {
    resolved
        .operands()
        .iter()
        .map(|(node, _, _)| packed_operands.get(node).copied())
        .collect()
}

pub fn emit(resolved: &BoundOp, packed_operands: &PackedOperands) -> Result<Kernel, EmitError> {
    validate(resolved)?;
    let entry = entry_name(resolved);
    let quantized = operand_codecs(resolved, packed_operands);
    let source = match &resolved.kind {
        BoundOpKind::Elementwise { .. } => render_elementwise(resolved, &entry, &quantized),
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => render_reduce(resolved, &entry, &quantized),
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => render_scan(resolved, &entry, &quantized),
        BoundOpKind::Iota => render_iota(resolved, &entry),
        BoundOpKind::Constant { value } => render_constant(resolved, &entry, *value),
    }?;
    Ok(Kernel {
        source,
        entry,
        bindings: bindings(resolved),
        grid: GridSpec {
            threads: grid_threads(resolved, &quantized),
            threadgroup_width: tiled_gemm_threadgroup_width(resolved, &quantized),
        },
    })
}

/// Cheap structural identity for the kernel [`emit`] would produce from
/// `resolved` — built without ever rendering the MSL body text, so a caller
/// can decide whether a pipeline compile is needed before paying for one.
/// Must distinguish anything [`emit`]'s `source` could differ on:
/// [`entry_name`] already carries rank / output-rank / operand-count / body /
/// reduce-op / keep / init / gather shape; this adds three axes `entry_name`
/// does NOT cover:
///
/// - [`type_token`]'s "half" vs "float" split — every dtype `emit` accepts
///   collapses to one of those two declarations.
/// - Per operand, which [`PackedCodec`] (if any) it reads through, AND
///   whether the op takes the row-blocked packed-matmul path
///   ([`packed_row_block`]) — that gate reads CONCRETE extents/strides, not
///   just op structure, so two ops agreeing on every field above can still
///   emit different bodies if only one of them clears it.
/// - For a `Reduce`, `output_axes`' own EXACT ORDERED sequence, not just its
///   length: `render_reduce`/`render_scan` bake the literal axis index tied
///   to each `output_extents`/`operand_strides` uniform slot straight into
///   the source text (e.g. `coord_q[{dim}] = ... u.output_extents[{index}]`),
///   so two folds sharing every field above but keeping a DIFFERENT axis SET
///   (or the same set in a different order) still emit different source.
///   `reduce_dims` needs no separate entry: it is `(0..rank)` minus
///   `output_axes` as a SET, always ascending, so `rank` + this exact
///   sequence already pins it down.
///
/// # Errors
/// Propagates [`type_token`]'s unsupported-dtype rejection — the same gate
/// [`emit`] enforces before ever building a kernel.
pub(crate) fn kernel_cache_key(resolved: &BoundOp, packed_operands: &PackedOperands) -> Result<String, EmitError> {
    let quantized = operand_codecs(resolved, packed_operands);
    let mut key = entry_name(resolved);
    key.push('_');
    key.push_str(type_token(resolved.node, resolved.dtype)?);
    for codec in &quantized {
        key.push(match codec {
            Some(PackedCodec::Q4K) => '4',
            Some(PackedCodec::Q5K) => '5',
            Some(PackedCodec::Q6K) => '6',
            None => 'f',
        });
    }
    // 'G' (tiled `simdgroup_matrix` GEMM) is checked FIRST: `tiled_gemm_block`
    // only ever returns `Some` when `packed_row_block` also would (it is
    // built ON TOP of that same gate), so the two are mutually exclusive by
    // construction and this order costs nothing extra to get right.
    key.push(
        if let BoundOpKind::Reduce {
            reduce_op,
            init,
            output_axes,
            ..
        } = &resolved.kind
        {
            if tiled_gemm_block(resolved, &quantized, *reduce_op, *init, output_axes).is_some() {
                'G'
            } else if packed_row_block(resolved, &quantized).is_some() {
                'B'
            } else {
                'S'
            }
        } else {
            'S'
        },
    );
    if let BoundOpKind::Reduce { output_axes, .. } = &resolved.kind {
        key.push_str("_ax");
        for axis in output_axes {
            key.push('_');
            key.push_str(&axis.to_string());
        }
    }
    Ok(key)
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
pub(crate) fn kernel_dispatch_shape(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
) -> Result<(Vec<Binding>, GridSpec), EmitError> {
    validate(resolved)?;
    let quantized = operand_codecs(resolved, packed_operands);
    Ok((
        bindings(resolved),
        GridSpec {
            threads: grid_threads(resolved, &quantized),
            threadgroup_width: tiled_gemm_threadgroup_width(resolved, &quantized),
        },
    ))
}

// `SIMD_WIDTH` moved to `crate::sized::SIMD_WIDTH` (the build-time floor's
// only configuration surface) -- imported at the top of this file.

/// Whether `resolved` is a `Keep::Reduce` fold whose `reduce_op` is
/// associative and commutative (`Add`, `Multiply`, `Maximum`, `Minimum`) with
/// no gathered operand — the set [`render_reduce`] emits a SIMD-group
/// cooperative loop for instead of the one-thread-per-output serial fold.
/// `Subtract`/`Divide` are not associative, so reordering their combination
/// across lanes is not imprecise, it is wrong — they and every other
/// `ScalarOp` stay on the serial path. Gather is excluded too: cooperative
/// striding would need each lane recording its own fault-slot contribution,
/// which this pass does not implement — default to serial when unsure.
fn reduce_is_cooperative(resolved: &BoundOp) -> bool {
    match &resolved.kind {
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            reduce_op,
            ..
        } => gather_count(resolved) == 0 && is_cooperative_reduce_op(*reduce_op),
        _ => false,
    }
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
fn simd_combine_fn(op: ScalarOp) -> &'static str {
    match op {
        ScalarOp::Add => "simd_sum",
        ScalarOp::Multiply => "simd_product",
        ScalarOp::Maximum => "simd_max",
        ScalarOp::Minimum => "simd_min",
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
        | ScalarOp::Select => unreachable!("simd_combine_fn is only called for a cooperative reduce_op"),
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
fn cooperative_identity_token(op: ScalarOp) -> &'static str {
    match op {
        ScalarOp::Add => "0.0f",
        ScalarOp::Multiply => "1.0f",
        ScalarOp::Maximum => "-INFINITY",
        ScalarOp::Minimum => "INFINITY",
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
        | ScalarOp::Select => unreachable!("cooperative_identity_token is only called for a cooperative reduce_op"),
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
        reduce_op, keep, ..
    } = &resolved.kind
    {
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
    let mut bindings: Vec<Binding> = resolved
        .operands()
        .iter()
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
}

/// Why a given [`BoundOp`] did NOT take the row-blocked packed kernel —
/// [`classify_packed_row_block`]'s error arm, one variant per gate in that
/// function's own condition order. `#[non_exhaustive]` so a new gate added
/// later is a compile error at every match site instead of a silently
/// unmatched `_`. Always compiled (not feature-gated itself) so
/// [`classify_packed_row_block`] — called from the unconditional emit path
/// — never needs a second copy of these seven conditions; only the public
/// accessor [`diagnose_packed_row_block`] is gated behind `instrument`, this
/// crate's diagnostic-only feature (see
/// [`crate::metal::execute_plan_op_timed`]'s own doc for why diagnostics
/// live behind that gate).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PackedRowBlockRejection {
    /// [`reduce_is_cooperative`] is false — not `Add`/`Multiply`/`Maximum`/
    /// `Minimum`, or the op gathers.
    NotCooperativeReduce,
    /// Not a `Reduce { keep: Keep::Reduce, .. }` at all.
    NotReduceKeepReduce,
    /// `quantized.len() != 2` — not a two-operand (weight, activation) op.
    OperandCountNotTwo,
    /// Neither exactly zero nor exactly one operand is packed.
    NotExactlyOnePackedOperand,
    /// The reduce folds ZERO axes into its output — degenerate, never
    /// observed on a real matmul (kept so the match stays exhaustive over
    /// every way [`reduce_dims`](reduction_dims) can come back empty).
    NotExactlyOneReduceDim { reduce_dims: Vec<u16> },
    /// More than one reduce dim, but they do NOT nest contiguously for both
    /// operands (see [`classify_packed_row_block`]'s own doc for the
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
            unreachable!("windows(2) always yields a two-element slice")
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
        ..
    } = &resolved.kind
    else {
        return Err(PackedRowBlockRejection::NotReduceKeepReduce);
    };
    if quantized.len() != 2 {
        return Err(PackedRowBlockRejection::OperandCountNotTwo);
    }
    let packed: Vec<usize> = quantized
        .iter()
        .enumerate()
        .filter_map(|(index, codec)| codec.is_some().then_some(index))
        .collect();
    let [weight] = packed[..] else {
        return Err(PackedRowBlockRejection::NotExactlyOnePackedOperand);
    };
    let Some(codec) = quantized[weight] else {
        unreachable!("weight index came from the is_some() filter above")
    };
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
    let extent: u64 = reduce_dims.iter().map(|&dim| resolved.extents[dim as usize]).product();
    if !(extent as usize).is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(PackedRowBlockRejection::ExtentNotBlockMultiple { extent });
    }
    Ok(PackedRowBlock {
        weight,
        other,
        reduce_dim: innermost as usize,
        codec,
    })
}

fn packed_row_block(resolved: &BoundOp, quantized: &[Option<PackedCodec>]) -> Option<PackedRowBlock> {
    classify_packed_row_block(resolved, quantized).ok()
}

/// Public diagnostic seam for [`classify_tiled_gemm`], same shape as
/// [`PackedRowBlockRejection`] for [`classify_packed_row_block`]: one variant
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
    /// [`classify_tiled_gemm`]'s own doc).
    FeatureDisabled,
    /// [`classify_packed_row_block`] itself rejected first; the tiled path
    /// can only narrow that gate's `Ok`, never rescue its `Err`.
    NotPackedRowBlock(PackedRowBlockRejection),
    /// The packed operand's codec is not [`PackedCodec::Q4K`] -- Q5_K/Q6_K
    /// have no batched-unpack helper for this path yet (see
    /// [`classify_tiled_gemm`]'s own comment).
    NotQ4K,
    /// `reduce_op`/`init` are not the plain `Add`-from-`Zero` shape
    /// `simdgroup_matrix` accumulation requires.
    NotAddZeroReduce,
    /// [`is_plain_product_reduce`] is false -- the fused body carries more
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
    /// output layout) — see [`axes_fold_contiguously`].
    AxisGroupNotContiguous,
    /// The token group's flattened extent is below
    /// [`crate::sized::TILED_GEMM_MIN_TOKENS`] -- tiling overhead is not
    /// amortized at this size.
    TokenExtentBelowMinimum { token_extent: u64, min_tokens: u64 },
}

/// The additional narrowing [`push_tiled_gemm_body`]'s `simdgroup_matrix`
/// path requires on top of [`packed_row_block`]'s own row-blocked
/// eligibility -- the one decision [`grid_threads`] and
/// [`push_cooperative_reduce_body`] must reach IDENTICALLY, same discipline
/// [`PackedRowBlock`] itself follows (see its own doc): this reads a
/// CONCRETE extent (the activation/token axis, against
/// [`crate::sized::TILED_GEMM_MIN_TOKENS`]) on top of `packed_row_block`'s
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
        } = classify_packed_row_block(resolved, quantized).map_err(TiledGemmRejection::NotPackedRowBlock)?;
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
            match (weight_layout.stride(axis) == 0, other_layout.stride(axis) == 0) {
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
        let reassembled: Vec<u16> = token_axes.iter().chain(feature_axes.iter()).copied().collect();
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
        let out_layout = match &resolved.kind {
            BoundOpKind::Reduce { out_layout, .. } => out_layout,
            _ => unreachable!("classify_packed_row_block above only matches Keep::Reduce"),
        };
        let groups_contiguous = axes_fold_contiguously(&token_axes, &resolved.extents, other_layout)
            && axes_fold_contiguously(&feature_axes, &resolved.extents, weight_layout)
            && axes_fold_contiguously(&token_axes, &resolved.extents, out_layout)
            && axes_fold_contiguously(&feature_axes, &resolved.extents, out_layout);
        if !groups_contiguous {
            return Err(TiledGemmRejection::AxisGroupNotContiguous);
        }
        let token_extent: u64 = token_axes.iter().map(|&axis| resolved.extents[axis as usize]).product();
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
fn tiled_gemm_threadgroups(feature_extent: u64, token_extent: u64) -> u64 {
    #[cfg(not(feature = "metal-tiled-gemm"))]
    {
        let _ = (feature_extent, token_extent);
        unreachable!("only called when tiled_gemm_block returned Some, which requires the feature")
    }
    #[cfg(feature = "metal-tiled-gemm")]
    {
        let row_tiles = feature_extent.div_ceil(crate::sized::TILED_GEMM_BLOCK_M);
        let col_tiles = token_extent.div_ceil(crate::sized::TILED_GEMM_BLOCK_N);
        row_tiles * col_tiles * (TILED_GEMM_NSG as u64) * SIMD_WIDTH
    }
}

fn grid_threads(resolved: &BoundOp, quantized: &[Option<PackedCodec>]) -> u64 {
    match &resolved.kind {
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
            if let Some(block) = tiled_gemm_block(resolved, quantized, *reduce_op, *init, output_axes) {
                // TILED_GEMM_NSG simdgroups per BLOCK_M x BLOCK_N output
                // tile, tiled over BOTH the feature axis and the token axis
                // — the amortization the row-blocked path does not do (it
                // tiles the feature axis alone; see `push_tiled_gemm_body`'s
                // doc).
                tiled_gemm_threadgroups(
                    block.feature_axes.iter().map(|&axis| resolved.extents[axis as usize]).product(),
                    block.token_axes.iter().map(|&axis| resolved.extents[axis as usize]).product(),
                )
            } else if packed_row_block(resolved, quantized).is_some() {
                // one SIMD group per PACKED_ROWS_PER_GROUP outputs
                output_total.div_ceil(PACKED_ROWS_PER_GROUP as u64) * SIMD_WIDTH
            } else if reduce_is_cooperative(resolved) {
                // one SIMD-group (SIMD_WIDTH lanes) per output element, not
                // one thread — see `reduce_is_cooperative`'s doc.
                output_total * SIMD_WIDTH
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
    }
}

fn op_token(op: ScalarOp) -> &'static str {
    match op {
        ScalarOp::Identity => "identity",
        ScalarOp::Add => "add",
        ScalarOp::Subtract => "subtract",
        ScalarOp::Multiply => "multiply",
        ScalarOp::Divide => "divide",
        ScalarOp::Maximum => "maximum",
        ScalarOp::Minimum => "minimum",
        ScalarOp::Negate => "negate",
        ScalarOp::Reciprocal => "reciprocal",
        ScalarOp::Exponential => "exponential",
        ScalarOp::Logarithm => "logarithm",
        ScalarOp::SquareRoot => "square_root",
        ScalarOp::Tanh => "tanh",
        ScalarOp::Erf => "erf",
        ScalarOp::Greater => "greater",
        ScalarOp::Equal => "equal",
        ScalarOp::Select => "select",
    }
}

fn init_token(init: ReduceInit) -> &'static str {
    match init {
        ReduceInit::Zero => "zero",
        ReduceInit::One => "one",
        ReduceInit::NegativeInfinity => "negative_infinity",
        ReduceInit::PositiveInfinity => "positive_infinity",
        ReduceInit::FirstElement => "first_element",
    }
}

fn keep_token(keep: Keep) -> &'static str {
    match keep {
        Keep::Reduce => "reduce",
        Keep::Scan => "scan",
    }
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
/// Whether `body` is the unfused, one-step, sequential-operand shape every
/// body had before fusion existed — the case [`body_token`] keeps naming
/// exactly as it always has, so every kernel name this crate emitted before
/// fusion existed is unchanged.
fn is_leaf(body: &ComposedBody) -> bool {
    body.steps.len() == 1
        && body.steps[0].args.iter().enumerate().all(
            |(index, arg)| matches!(arg, StepArg::Operand(operand) if *operand as usize == index),
        )
}

/// A valid-MSL-identifier fingerprint of every step in a fused body: which
/// op, over which operand slots or earlier steps, in order — two bodies with
/// the same structure (independent of concrete extents/strides/buffers)
/// must fingerprint identically so the kernel they emit is cacheable by
/// structure, matching this module's own stance on `entry_name` overall.
fn body_fingerprint(body: &ComposedBody) -> String {
    body.steps
        .iter()
        .map(|step| {
            let mut token = String::from(op_token(step.op));
            for arg in &step.args {
                match arg {
                    StepArg::Operand(index) => token.push_str(&format!("_o{index}")),
                    StepArg::Step(index) => token.push_str(&format!("_s{index}")),
                }
            }
            token
        })
        .collect::<Vec<_>>()
        .join("__")
}

fn body_token(body: &ComposedBody) -> String {
    if is_leaf(body) {
        op_token(body.steps[0].op).into()
    } else {
        format!("fused_{}", body_fingerprint(body))
    }
}

fn entry_name(resolved: &BoundOp) -> String {
    let rank = resolved.extents.len();
    let operand_count = resolved.operands().len();
    let base = match &resolved.kind {
        BoundOpKind::Elementwise { .. } => {
            let body = body_token(resolved.element_body());
            format!("omega_elementwise_r{rank}_n{operand_count}_{body}")
        }
        BoundOpKind::Reduce {
            reduce_op,
            init,
            keep,
            output_axes,
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
            format!("omega_{kind}_r{rank}_o{output_rank}_n{operand_count}_{body}_{reduce_body}_{init}")
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
    for (index, step) in body.steps.iter().enumerate() {
        let args: Vec<String> = step
            .args
            .iter()
            .map(|arg| match arg {
                StepArg::Operand(operand_index) => format!("scratch[{operand_index}]"),
                StepArg::Step(step_index) => format!("step{step_index}"),
            })
            .collect();
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let expr = scalar_op_expr(step.op, &arg_refs);
        source.push_str(&format!("{indent}{element_type} step{index} = {expr};\n"));
    }
    format!("step{}", body.steps.len().saturating_sub(1))
}

fn kernel_signature(
    source: &mut String,
    quantized: &[Option<PackedCodec>],
    gather_count: usize,
    entry: &str,
    element_type: &str,
) {
    let operand_count = quantized.len();
    source.push_str(&format!("kernel void {entry}(\n"));
    for (index, &codec) in quantized.iter().enumerate() {
        // a packed operand's buffer is BYTES, not elements — the shader
        // turns an element offset into a super-block plus a position inside
        // it at the read (`operand_read`), so the binding has to be typed
        // for what is actually in the buffer.
        let binding_type = if codec.is_some() { "uchar" } else { element_type };
        source.push_str(&format!(
            "    device const {binding_type}* in{index} [[buffer({index})]],\n"
        ));
    }
    for slot in 0..gather_count {
        // a gather's fetched index is always carried as an exact-integer
        // `float`, independent of the op's own element type — see this
        // crate's doc for `gather_idx` and `cpu::reject_non_float32`'s own
        // note on indices being the one deliberate non-dtype exception.
        source.push_str(&format!(
            "    device const float* gather_idx{slot} [[buffer({})]],\n",
            operand_count + slot
        ));
    }
    source.push_str(&format!(
        "    device {element_type}* out [[buffer({})]],\n",
        operand_count + gather_count
    ));
    source.push_str(&format!(
        "    constant Uniforms& u [[buffer({})]],\n",
        operand_count + gather_count + 1
    ));
    if gather_count > 0 {
        source.push_str(&format!(
            "    device atomic_uint* fault [[buffer({})]],\n",
            operand_count + gather_count + 2
        ));
    }
    source.push_str("    uint gid [[thread_position_in_grid]])\n{\n");
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
    // emitted unconditionally, the same way `PROXIMA_ERF_FN` is: a
    // `static inline` the kernel never calls costs nothing in the compiled
    // AIR, and making it conditional would mean threading "does this kernel
    // read a packed operand" into the preamble for no gain.
    source.push_str(Q4K_UNPACK_MSL);
    source.push('\n');
    source.push_str(Q5K_UNPACK_MSL);
    source.push('\n');
    source.push_str(Q6K_UNPACK_MSL);
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
        Some(PackedCodec::Q4K) => format!(
            "q4k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q4K_BLOCK_BYTES}, (uint)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        ),
        Some(PackedCodec::Q5K) => format!(
            "q5k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q5K_BLOCK_BYTES}, (uint)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        ),
        Some(PackedCodec::Q6K) => format!(
            "q6k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q6K_BLOCK_BYTES}, (uint)({offset} % {Q4K_BLOCK_ELEMENTS}))"
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

    kernel_signature(&mut source, &[], 0, entry, element_type);
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

    kernel_signature(&mut source, &[], 0, entry, element_type);
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

fn render_elementwise(resolved: &BoundOp, entry: &str, quantized: &[Option<PackedCodec>]) -> Result<String, EmitError> {
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    let gather_count = gather_count(resolved);
    let gather_slots = gather_slots(resolved);
    let element_type = type_token(resolved.node, resolved.dtype)?;

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

    kernel_signature(&mut source, quantized, gather_count, entry, element_type);
    source.push_str("    if ((long)gid >= u.total_elements) { return; }\n");

    if rank > 0 {
        source.push_str(&format!("    long coord[{rank_len}];\n"));
        source.push_str("    long remaining = (long)gid;\n");
        for dim in (0..rank).rev() {
            source.push_str(&format!(
                "    coord[{dim}] = remaining % u.extents[{dim}]; remaining /= u.extents[{dim}];\n"
            ));
        }
    }

    for (index, gather_slot) in gather_slots.iter().enumerate() {
        source.push_str(&format!("    long off{index} = u.operand_base[{index}];\n"));
        for dim in 0..rank {
            source.push_str(&format!(
                "    off{index} += coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
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

fn render_reduce(resolved: &BoundOp, entry: &str, quantized: &[Option<PackedCodec>]) -> Result<String, EmitError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        ..
    } = &resolved.kind
    else {
        unreachable!("render_reduce is only called for a Keep::Reduce fold")
    };
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    let output_rank = output_axes.len();
    let output_rank_len = output_rank.max(1);
    let reduce_dims = reduction_dims(resolved, output_axes);
    let reduce_rank = reduce_dims.len();
    let reduce_rank_len = reduce_rank.max(1);
    let gather_count = gather_count(resolved);
    let gather_slots = gather_slots(resolved);
    let element_type = type_token(resolved.node, resolved.dtype)?;

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
    push_gather_uniform_fields(&mut source, gather_count, rank_len);
    source.push_str("};\n\n");

    kernel_signature(&mut source, quantized, gather_count, entry, element_type);

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
        );
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
        );
    }
    source.push_str("}\n");
    Ok(source)
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
    source.push_str("    out[out_offset] = accumulator;\n");
}

/// The SIMD-group cooperative fold: `SIMD_WIDTH` lanes split one output
/// element's contraction axis, each striding through `reduction_total` by
/// `SIMD_WIDTH` so every element is visited by exactly one lane, then
/// combine via [`simd_combine_fn`]. Only lane 0 writes the result, and only
/// lane 0 seeds from the `BoundOp`'s real `ReduceInit` — every other lane
/// seeds from [`cooperative_identity_token`] so the true seed is folded into
/// the group exactly once (see that function's doc). `gid / SIMD_WIDTH` is a
/// valid output index, and `gid % SIMD_WIDTH` a valid lane-within-group
/// index, only because [`GridSpec::threadgroup_width`] pins the dispatched
/// threadgroup width to exactly `SIMD_WIDTH` — see `crate::metal::dispatch`.
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
fn is_plain_product_reduce(resolved: &BoundOp, reduce_op: ScalarOp, weight: usize, other: usize) -> bool {
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
) {
    let Some(block) = packed_row_block(resolved, quantized) else {
        unreachable!("push_packed_row_blocked_body is only called when packed_row_block matched")
    };
    let PackedRowBlock {
        weight,
        other,
        reduce_dim,
        codec,
    } = block;
    let block_bytes = codec.block_bytes();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    // seeded on lane 0 only, exactly as the general cooperative path does:
    // the true seed folds in once and every other lane starts at the
    // algebraic identity, so `simd_*` can combine them unconditionally.
    let (init_expr, _) = fold_init_tokens(init);
    let identity = cooperative_identity_token(reduce_op);
    // ROW-BLOCKED PACKED PATH. One SIMD group folds PACKED_ROWS_PER_GROUP
    // output rows at once so the activation's run of 8 values is loaded into
    // registers ONCE and reused across all of them — ggml's `float
    // sumf[nr0]` with `N_R0_Q4_K 4`. Combined with the super-block header
    // amortization below, the per-element cost becomes one byte load, one
    // mask, one fma (`docs/discipline.md` ROW 74).
    {
        let run = Q4K_BLOCK_ELEMENTS / SIMD_WIDTH as usize;
        let rows = PACKED_ROWS_PER_GROUP;
        source.push_str(&format!("    long group_first = output_index * {rows};\n"));
        source.push_str(&format!("    {element_type} sumf[{rows}];\n"));
        source.push_str(&format!(
            "    for (int q = 0; q < {rows}; ++q) {{ sumf[q] = (lane == 0u) ? ({init_expr}) : ({identity}); }}\n"
        ));
        source.push_str(&format!("    long weight_base[{rows}];\n"));
        source.push_str(&format!("    long other_base[{rows}];\n"));
        source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
        source.push_str("        long flat = group_first + q;\n");
        source.push_str("        long remaining_q = flat;\n");
        source.push_str(&format!("        long coord_q[{rank_len}];\n"));
        source.push_str(&format!("        for (int d = 0; d < {rank}; ++d) {{ coord_q[d] = 0; }}\n"));
        for (index, dim) in output_axes.iter().enumerate().rev() {
            source.push_str(&format!(
                "        coord_q[{dim}] = remaining_q % u.output_extents[{index}]; remaining_q /= u.output_extents[{index}];\n"
            ));
        }
        source.push_str(&format!("        long wb = u.operand_base[{weight}];\n"));
        source.push_str(&format!("        long ob = u.operand_base[{other}];\n"));
        // Iterate the OUTPUT axes directly rather than `0..rank` minus one
        // excluded dim: `reduce_dim` is now the innermost of possibly
        // SEVERAL folded reduce dims (see `classify_packed_row_block`'s
        // contiguous-fold check), so `output_axes` — already the exact
        // complement of every reduce dim, however many there are — is the
        // correct and simpler set to walk here regardless of reduce rank.
        for &dim in output_axes {
            source.push_str(&format!(
                "        wb += coord_q[{dim}] * u.operand_strides[{weight}][{dim}];\n"
            ));
            source.push_str(&format!(
                "        ob += coord_q[{dim}] * u.operand_strides[{other}][{dim}];\n"
            ));
        }
        source.push_str("        weight_base[q] = wb;\n");
        source.push_str("        other_base[q] = ob;\n");
        source.push_str("    }\n");
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
        source.push_str(&format!("    uint ix = (uint)lane / {lanes_per_block}u;\n"));
        source.push_str(&format!("    uint it = (uint)lane % {lanes_per_block}u;\n"));
        source.push_str(&format!("    uint slot = it * {sub}u;\n"));
        source.push_str(&format!("    {element_type} acts[{sub}];\n"));
        source.push_str(&format!(
            "    int super_blocks = (int)u.reduction_total / {Q4K_BLOCK_ELEMENTS};\n"
        ));
        source.push_str(&format!(
            "    for (int ib = (int)ix; ib < super_blocks; ib += {}) {{\n",
            SIMD_WIDTH as usize / lanes_per_block
        ));
        source.push_str(&format!(
            "        int elem0 = ib * {Q4K_BLOCK_ELEMENTS} + (int)slot;\n"
        ));
        source.push_str(&format!("        for (int j = 0; j < {sub}; ++j) {{\n"));
        source.push_str(&format!(
            "            acts[j] = in{other}[other_base[0] + (long)(elem0 + j) * other_stride];\n"
        ));
        source.push_str("        }\n");
        source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
        source.push_str(&format!(
            "            device const uchar *blk = in{weight} + ((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS} + ib) * {block_bytes};\n"
        ));
        match codec {
            PackedCodec::Q4K => {
                source.push_str("            q4k_header hdr = q4k_header_for(blk, slot);\n");
                if is_plain_product_reduce(resolved, reduce_op, weight, other) {
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
                    source.push_str(&format!("            {element_type} raw_acc = 0;\n"));
                    source.push_str(&format!("            {element_type} act_sum = 0;\n"));
                    source.push_str(&format!("            for (int c = 0; c < {}; ++c) {{\n", sub / run));
                    // raw 4-bit levels (0..15) are exact in float regardless
                    // of the kernel's element type; q4k_run8 takes `thread
                    // float *out`, narrowed to element_type at the multiply
                    // below, same as the per-element path.
                    source.push_str(&format!("                float levels[{run}];\n"));
                    source.push_str(&format!(
                        "                q4k_run8(blk, slot + (uint)(c * {run}), levels);\n"
                    ));
                    source.push_str(&format!("                for (int j = 0; j < {run}; ++j) {{\n"));
                    source.push_str(&format!(
                        "                    {element_type} act = acts[c * {run} + j];\n"
                    ));
                    source.push_str("                    raw_acc += levels[j] * act;\n");
                    source.push_str("                    act_sum += act;\n");
                    source.push_str("                }\n");
                    source.push_str("            }\n");
                    source.push_str(
                        "            sumf[q] = sumf[q] + hdr.scale * raw_acc - hdr.minimum * act_sum;\n",
                    );
                } else {
                    source.push_str(&format!("            for (int c = 0; c < {}; ++c) {{\n", sub / run));
                    // raw 4-bit levels (0..15) are exact in float regardless of
                    // the kernel's element type; q4k_run8 takes `thread float
                    // *out`, and the narrowing to element_type happens where
                    // levels combine into scratch below, same as every other
                    // operand read.
                    source.push_str(&format!("                float levels[{run}];\n"));
                    source.push_str(&format!(
                        "                q4k_run8(blk, slot + (uint)(c * {run}), levels);\n"
                    ));
                    source.push_str(&format!("                for (int j = 0; j < {run}; ++j) {{\n"));
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
                    let value_expr =
                        push_body_steps(source, resolved.element_body(), "                    ", element_type);
                    source.push_str(&format!("                    {element_type} value = {value_expr};\n"));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                    sumf[q] = {combine_expr};\n"));
                    source.push_str("                }\n");
                    source.push_str("            }\n");
                }
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
                // 92) for the measured cost of skipping it.
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
                let value_expr =
                    push_body_steps(source, resolved.element_body(), "                ", element_type);
                source.push_str(&format!("                {element_type} value = {value_expr};\n"));
                let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                source.push_str("            }\n");
            }
            PackedCodec::Q6K => {
                // No `q6k_run8`-style batched unpack yet — `Q6_K`'s bit
                // layout does not reduce to two word loads the way `Q4_K`'s
                // does (each element needs a `ql` byte, a `qh` byte, AND a
                // sub-block scale byte, not one nibble out of an
                // already-loaded word). Correct, one element at a time; `d`
                // is still decoded ONCE per super-block via
                // `q6k_header_for` rather than per element. A follow-up
                // optimization, not a correctness gap — see this landing's
                // discipline row for the measured cost of skipping it.
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
                let value_expr =
                    push_body_steps(source, resolved.element_body(), "                ", element_type);
                source.push_str(&format!("                {element_type} value = {value_expr};\n"));
                let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                source.push_str("            }\n");
            }
        }
        source.push_str("        }\n");
        source.push_str("    }\n");
        let combine_fn = simd_combine_fn(reduce_op);
        source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
        source.push_str(&format!(
            "        {element_type} reduced = {combine_fn}(sumf[q]);\n"
        ));
        source.push_str("        long flat = group_first + q;\n");
        source.push_str("        if (lane == 0u && flat < u.output_total) {\n");
        source.push_str("            long remaining_q = flat;\n");
        source.push_str(&format!("            long coord_q[{rank_len}];\n"));
        source.push_str(&format!("            for (int d = 0; d < {rank}; ++d) {{ coord_q[d] = 0; }}\n"));
        for (index, dim) in output_axes.iter().enumerate().rev() {
            source.push_str(&format!(
                "            coord_q[{dim}] = remaining_q % u.output_extents[{index}]; remaining_q /= u.output_extents[{index}];\n"
            ));
        }
        source.push_str("            long out_offset = u.out_base;\n");
        for dim in 0..rank {
            source.push_str(&format!(
                "            out_offset += coord_q[{dim}] * u.out_strides[{dim}];\n"
            ));
        }
        source.push_str("            out[out_offset] = reduced;\n");
        source.push_str("        }\n");
        source.push_str("    }\n");
    }
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
    output_axes: &[u16],
    rank: usize,
    block: &TiledGemmBlock,
    element_type: &str,
) {
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
        unreachable!("classify_tiled_gemm never builds an empty token group")
    };
    let Some(&feature_axis) = feature_axes.last() else {
        unreachable!("classify_tiled_gemm never builds an empty feature group")
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
    let group_extent_expr = |group: &[u16]| -> String {
        group
            .iter()
            .map(|&dim| {
                let Some(index) = output_axes.iter().position(|&candidate| candidate == dim) else {
                    unreachable!("every token/feature axis is one of output_axes by construction")
                };
                format!("u.output_extents[{index}]")
            })
            .collect::<Vec<_>>()
            .join(" * ")
    };

    source.push_str(&format!(
        "    long feature_extent = {};\n",
        group_extent_expr(feature_axes)
    ));
    source.push_str(&format!(
        "    long token_extent = {};\n",
        group_extent_expr(token_axes)
    ));
    source.push_str(&format!(
        "    long num_col_tiles = (token_extent + {}) / {block_n};\n",
        block_n - 1
    ));
    source.push_str(&format!("    long tiitg = (long)gid % {block_threads};\n"));
    source.push_str(&format!("    long sgitg = tiitg / {SIMD_WIDTH};\n"));
    source.push_str(&format!("    long tile_index = (long)gid / {block_threads};\n"));
    source.push_str("    long row_tile = tile_index / num_col_tiles;\n");
    source.push_str("    long col_tile = tile_index % num_col_tiles;\n");
    source.push_str("    long row_half = sgitg & 1;\n");
    source.push_str("    long col_half = sgitg >> 1;\n");
    source.push_str(&format!("    threadgroup half weight_tile[{weight_tile_elems}];\n"));
    source.push_str(&format!("    threadgroup float act_tile[{act_tile_elems}];\n"));
    source.push_str(&format!("    simdgroup_float8x8 acc[{mc_count}];\n"));
    source.push_str(&format!(
        "    for (int i = 0; i < {mc_count}; ++i) {{ acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f); }}\n"
    ));
    source.push_str(&format!("    for (long k0 = 0; k0 < u.reduction_total; k0 += {block_k}) {{\n"));
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
    source.push_str(&format!("            long w_feat = row_tile * {block_m} + w_row;\n"));
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
    source.push_str(&format!("            long a_tok = col_tile * {block_n} + a_col;\n"));
    source.push_str("            long a_k_global = k0 + a_k;\n");
    source.push_str("            float a_value = 0.0f;\n");
    source.push_str("            if (a_tok < token_extent) {\n");
    source.push_str(&format!(
        "                long aoff = u.operand_base[{other}] + a_tok * u.operand_strides[{other}][{token_axis}] + a_k_global * u.operand_strides[{other}][{reduce_dim}];\n"
    ));
    source.push_str(&format!("                a_value = {};\n", operand_read(other, "aoff", None)));
    source.push_str("            }\n");
    source.push_str(&format!("            act_tile[a_col * {block_k} + a_k] = a_value;\n"));
    source.push_str("        }\n");
    source.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str(&format!("        for (int sub_k = 0; sub_k < {sub_k_steps}; ++sub_k) {{\n"));
    source.push_str(&format!("            simdgroup_half8x8 a_frag[{thread_mat_m}];\n"));
    source.push_str(&format!("            for (int i = 0; i < {thread_mat_m}; ++i) {{\n"));
    source.push_str(&format!(
        "                simdgroup_load(a_frag[i], weight_tile + (row_half * {thread_mat_m} + i) * 8 * {block_k} + sub_k * 8, {block_k});\n"
    ));
    source.push_str("            }\n");
    source.push_str("            simdgroup_barrier(mem_flags::mem_none);\n");
    source.push_str(&format!("            simdgroup_float8x8 b_frag[{thread_mat_n}];\n"));
    source.push_str(&format!("            for (int j = 0; j < {thread_mat_n}; ++j) {{\n"));
    source.push_str(&format!(
        "                simdgroup_load(b_frag[j], act_tile + (col_half * {thread_mat_n} + j) * 8 * {block_k} + sub_k * 8, {block_k}, ulong2(0), true);\n"
    ));
    source.push_str("            }\n");
    source.push_str(&format!("            for (int i = 0; i < {thread_mat_m}; ++i) {{\n"));
    source.push_str(&format!("                for (int j = 0; j < {thread_mat_n}; ++j) {{\n"));
    source.push_str(&format!(
        "                    simdgroup_multiply_accumulate(acc[i * {thread_mat_n} + j], a_frag[i], b_frag[j], acc[i * {thread_mat_n} + j]);\n"
    ));
    source.push_str("                }\n");
    source.push_str("            }\n");
    source.push_str("        }\n");
    source.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("    }\n");
    source.push_str(&format!("    threadgroup float out_tile[{out_tile_elems}];\n"));
    source.push_str(&format!("    for (int i = 0; i < {thread_mat_m}; ++i) {{\n"));
    source.push_str(&format!("        for (int j = 0; j < {thread_mat_n}; ++j) {{\n"));
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
    source.push_str(&format!("        long o_feat = row_tile * {block_m} + o_row;\n"));
    source.push_str(&format!("        long o_tok = col_tile * {block_n} + o_col;\n"));
    source.push_str("        if (o_feat < feature_extent && o_tok < token_extent) {\n");
    source.push_str(&format!("            long coord[{rank_len}];\n"));
    source.push_str(&format!("            for (int d = 0; d < {rank}; ++d) {{ coord[d] = 0; }}\n"));
    source.push_str(&format!("            coord[{feature_axis}] = o_feat;\n"));
    source.push_str(&format!("            coord[{token_axis}] = o_tok;\n"));
    source.push_str("            long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!("            out_offset += coord[{dim}] * u.out_strides[{dim}];\n"));
    }
    source.push_str(&format!(
        "            out[out_offset] = ({element_type})out_tile[idx];\n"
    ));
    source.push_str("        }\n");
    source.push_str("    }\n");
}

/// Never actually invoked: [`classify_tiled_gemm`]'s own `#[cfg(not(feature
/// = "metal-tiled-gemm"))]` arm always returns `None`, so no caller ever
/// holds a `&TiledGemmBlock` to pass here without the feature -- this stub
/// exists only so [`push_cooperative_reduce_body`]'s `if let Some(block) =
/// tiled_gemm_block(...)` arm still type-checks in that build.
#[cfg(not(feature = "metal-tiled-gemm"))]
fn push_tiled_gemm_body(
    source: &mut String,
    output_axes: &[u16],
    rank: usize,
    block: &TiledGemmBlock,
    element_type: &str,
) {
    let _ = (source, output_axes, rank, block, element_type);
    unreachable!("classify_tiled_gemm only ever returns Some behind feature = \"metal-tiled-gemm\"")
}

/// The threadgroup width [`emit`]/[`kernel_dispatch_shape`] must dispatch
/// with -- [`TILED_GEMM_NSG`]` * SIMD_WIDTH` (128) when `resolved` takes
/// [`push_tiled_gemm_body`]'s multi-simdgroup path (its coordinate math
/// depends on exactly this many threads per threadgroup, the same
/// correctness requirement `crate::metal::dispatch`'s own doc states for
/// `SIMD_WIDTH`), `SIMD_WIDTH` for every other cooperative-reduce kernel,
/// `None` otherwise. Single source of truth both dispatch-shape functions
/// read, so they cannot drift the way two independent copies of this
/// `if`/`else` could.
fn tiled_gemm_threadgroup_width(resolved: &BoundOp, quantized: &[Option<PackedCodec>]) -> Option<u64> {
    if let BoundOpKind::Reduce {
        keep: Keep::Reduce,
        reduce_op,
        init,
        output_axes,
        ..
    } = &resolved.kind
        && tiled_gemm_block(resolved, quantized, *reduce_op, *init, output_axes).is_some()
    {
        return Some((TILED_GEMM_NSG as u64) * SIMD_WIDTH);
    }
    reduce_is_cooperative(resolved).then_some(SIMD_WIDTH)
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
) {
    let rank_len = rank.max(1);
    let output_rank = output_axes.len();
    let output_rank_len = output_rank.max(1);
    let reduce_rank = reduce_dims.len();
    let reduce_rank_len = reduce_rank.max(1);
    let operand_count = resolved.operands().len();

    // the tiled GEMM path owns its own preamble entirely (`tiitg`/`sgitg`/
    // `tile_index`, derived straight from `gid` against `TILED_GEMM_NSG *
    // SIMD_WIDTH` threads per threadgroup, ROW 109) -- it needs neither
    // `output_index` nor `lane` the way the row-blocked path below does, see
    // `kernel_cache_key`'s own comment for why the two are mutually
    // exclusive by construction.
    if let Some(block) = tiled_gemm_block(resolved, quantized, reduce_op, init, output_axes) {
        push_tiled_gemm_body(source, output_axes, rank, &block, element_type);
        return;
    }

    // the row-blocked packed path owns its own preamble: `output_index` is a
    // GROUP index there, not an output index, so the guard below would be
    // wrong for it.
    if packed_row_block(resolved, quantized).is_some() {
        source.push_str(&format!("    long output_index = (long)gid / {SIMD_WIDTH};\n"));
        source.push_str(&format!("    uint lane = gid % {SIMD_WIDTH}u;\n"));
        push_packed_row_blocked_body(
            source,
            resolved,
            reduce_op,
            init,
            output_axes,
            rank,
            quantized,
            element_type,
        );
        return;
    }

    source.push_str(&format!("    long output_index = (long)gid / {SIMD_WIDTH};\n"));
    source.push_str("    if (output_index >= u.output_total) { return; }\n");
    source.push_str(&format!("    uint lane = gid % {SIMD_WIDTH}u;\n"));

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
    let identity = cooperative_identity_token(reduce_op);
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
    let reduce_extent = resolved.extents[reduce_dim] as usize;
    let run = Q4K_BLOCK_ELEMENTS / SIMD_WIDTH as usize;
    let tiled = packed.len() == 1
        && resolved.operands()[packed[0]].1.stride(reduce_dims[0]) == 1
        && reduce_extent.is_multiple_of(Q4K_BLOCK_ELEMENTS);
    if tiled {
        let weight = packed[0];
        for index in 0..operand_count {
            source.push_str(&format!("    long base{index} = u.operand_base[{index}];\n"));
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
        let value_expr = push_body_steps(source, resolved.element_body(), "            ", element_type);
        source.push_str(&format!("            {element_type} value = {value_expr};\n"));
        let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
        source.push_str(&format!(
            "            accumulator = seeded ? {combine_expr} : value;\n"
        ));
        source.push_str("            seeded = true;\n");
        source.push_str("        }\n");
        source.push_str("    }\n");
        push_cooperative_reduce_tail(source, resolved, reduce_op, rank, element_type);
        return;
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
            source.push_str(&format!("    off{index} += (long)lane * stride{index};\n"));
            // 32-bit from here down. The offsets ABOVE stay `long` because a
            // layout base can legitimately be one; the per-element WALK never
            // needs that range, and Apple GPUs are 32-bit machines where
            // 64-bit integer arithmetic is emulated. `u.walk_fits_int` is the
            // runtime guard — when an operand's span really does exceed
            // `int`, the 64-bit walk below runs instead.
            source.push_str(&format!("    int walk{index} = (int)off{index};\n"));
            source.push_str(&format!(
                "    int advance{index} = (int)(stride{index} * {SIMD_WIDTH});\n"
            ));
        }
        source.push_str(&format!(
            "    for (int r = (int)lane; r < (int)u.reduction_total; r += {SIMD_WIDTH}) {{\n"
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
        push_cooperative_reduce_tail(source, resolved, reduce_op, rank, element_type);
        return;
    }

    source.push_str(&format!(
        "    for (long r = (long)lane; r < u.reduction_total; r += {SIMD_WIDTH}) {{\n"
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

    push_cooperative_reduce_tail(source, resolved, reduce_op, rank, element_type);
}

/// The `simd_sum` fold and the lane-0 store both cooperative loop shapes
/// end with — shared so the strength-reduced single-reduction-dim path and
/// the general path cannot drift on how the result is written out.
fn push_cooperative_reduce_tail(
    source: &mut String,
    _resolved: &BoundOp,
    reduce_op: ScalarOp,
    rank: usize,
    element_type: &str,
) {
    let combine_fn = simd_combine_fn(reduce_op);
    source.push_str(&format!(
        "    {element_type} reduced = {combine_fn}(accumulator);\n"
    ));
    source.push_str("    if (lane == 0u) {\n");
    source.push_str("        long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "        out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    source.push_str("        out[out_offset] = reduced;\n");
    source.push_str("    }\n");
}

fn render_scan(resolved: &BoundOp, entry: &str, quantized: &[Option<PackedCodec>]) -> Result<String, EmitError> {
    let BoundOpKind::Reduce {
        reduce_op, init, ..
    } = &resolved.kind
    else {
        unreachable!("render_scan is only called for a Keep::Scan fold")
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

    kernel_signature(&mut source, quantized, gather_count, entry, element_type);
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
    use alloc::vec;
    use alloc::vec::Vec;

    use proxima_tensor::{
        AxisTerm, DType, Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp, append, bind,
        infer, map,
    };

    use super::*;

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
        bind(&program, &shapes, &[])
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
        bind(&program, &shapes, &[])
            .expect("matmul lowers")
            .into_iter()
            .next()
            .expect("one fused bound emitted")
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
        bind(&program, &shapes, &[])
            .expect("matmul lowers")
            .into_iter()
            .next()
            .expect("one fused bound emitted")
    }

    #[test]
    fn q4k_row_blocked_matmul_defers_scale_to_once_per_sub_block() {
        // 256 == Q4K_BLOCK_ELEMENTS exactly: one super-block, so
        // packed_row_block matches and this is the real matmul shape the
        // scale-deferred path exists for (`docs/discipline.md` ROW 106).
        let bound = matmul_op(4, 256, 5);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);

        assert!(
            packed_row_block(&bound, &operand_codecs(&bound, &q4k)).is_some(),
            "test fixture must actually take the row-blocked path for this assertion to mean anything"
        );

        let source = emit(&bound, &q4k).expect("emits").source;
        assert!(
            source.contains("raw_acc"),
            "Add-reduce over a plain weight*activation body must take the scale-deferred path:\n{source}"
        );
        assert!(
            !source.contains("hdr.scale * levels[j] - hdr.minimum"),
            "the per-element dequant expression must not remain once the scale-deferred path is taken:\n{source}"
        );
    }

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

        let source = emit(&bound, &q4k).expect("emits").source;
        assert!(
            !source.contains("raw_acc"),
            "a Maximum reduce must never take the scale-deferred path, its identity does not hold under max:\n{source}"
        );
        assert!(
            source.contains("hdr.scale * levels[j] - hdr.minimum"),
            "a Maximum reduce must keep dequantizing per element:\n{source}"
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
        bind(&program, &shapes, &[])
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
                shape: vec![Extent::Static(embed), Extent::Static(heads), Extent::Static(head_dim)],
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
        bind(&program, &shapes, &[])
            .expect("multi-head matmul lowers")
            .into_iter()
            .next()
            .expect("one fused bound emitted")
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

        let source = emit(&bound, &q4k).expect("emits").source;
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
            tiled_gemm_block(&bound, &operand_codecs(&bound, &q4k), ScalarOp::Add, ReduceInit::Zero, &[1, 0]).is_none(),
            "one token must never clear TILED_GEMM_MIN_TOKENS"
        );
        let source = emit(&bound, &q4k).expect("emits").source;
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
            tiled_gemm_block(&bound, &operand_codecs(&bound, &q4k), ScalarOp::Add, ReduceInit::Zero, &[1, 0]).is_some(),
            "16 tokens must clear TILED_GEMM_MIN_TOKENS"
        );
        let source = emit(&bound, &q4k).expect("emits").source;
        assert!(
            source.contains("simdgroup_multiply_accumulate"),
            "a 16-token dispatch must take the tiled GEMM path:\n{source}"
        );
        assert!(source.contains("simdgroup_load"), "the tiled path must stage both operand tiles:\n{source}");
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
            tiled_gemm_block(&bound, &operand_codecs(&bound, &q6k), ScalarOp::Add, ReduceInit::Zero, &[1, 0]).is_none(),
            "a Q6_K weight must never take the tiled GEMM path"
        );
        let source = emit(&bound, &q6k).expect("emits").source;
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
        let source = emit(&bound, &q4k).expect("emits").source;
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
        bind(&program, &shapes, &[])
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
        bind(&program, &shapes, &[])
            .expect("embedding lookup lowers")
            .into_iter()
            .next()
            .expect("one bound emitted")
    }

    #[test]
    fn a_gather_op_emits_an_indices_binding_and_the_fetch_uniforms() {
        let bound = embedding_lookup_op(50_000, 8, 4);
        let kernel = emit(&bound, &BTreeMap::new()).expect("gather emits");

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
    fn a_gather_kernel_binds_and_declares_the_fault_buffer() {
        let bound = embedding_lookup_op(50_000, 8, 4);
        let kernel = emit(&bound, &BTreeMap::new()).expect("gather emits");

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
        let kernel = emit(&bound, &BTreeMap::new()).expect("gather-free elementwise emits");
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
        let kernel = emit(&bound, &BTreeMap::new()).expect("elementwise emits");

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
        bind(&program, &shapes, &[])
            .expect("rank3 identity sum lowers")
            .into_iter()
            .next()
            .expect("one bound emitted")
    }

    #[test]
    fn distinct_output_rank_at_same_total_rank_yields_distinct_cache_keys_and_source() {
        let keeps_two_axes = rank3_identity_sum_op(&[0, 1]);
        let keeps_one_axis = rank3_identity_sum_op(&[0]);

        assert_eq!(keeps_two_axes.extents.len(), keeps_one_axis.extents.len(), "same total rank");
        assert_eq!(
            keeps_two_axes.operands().len(),
            keeps_one_axis.operands().len(),
            "same operand count"
        );

        let empty = BTreeMap::new();
        let key_two_axes = kernel_cache_key(&keeps_two_axes, &empty).expect("cache key builds");
        let key_one_axis = kernel_cache_key(&keeps_one_axis, &empty).expect("cache key builds");
        assert_ne!(
            key_two_axes, key_one_axis,
            "a coarser key would let a 1-output-axis fold hit the 2-output-axis pipeline"
        );

        let source_two_axes = emit(&keeps_two_axes, &empty).expect("emits").source;
        let source_one_axis = emit(&keeps_one_axis, &empty).expect("emits").source;
        assert_ne!(
            source_two_axes, source_one_axis,
            "output_extents/reduction_extents array sizes must differ in the rendered source"
        );
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
        let key_first_second = kernel_cache_key(&keeps_first_and_second, &empty).expect("cache key builds");
        let key_first_third = kernel_cache_key(&keeps_first_and_third, &empty).expect("cache key builds");
        assert_ne!(
            key_first_second, key_first_third,
            "output_axes.len() alone cannot tell {{0,1}} from {{0,2}}"
        );

        let source_first_second = emit(&keeps_first_and_second, &empty).expect("emits").source;
        let source_first_third = emit(&keeps_first_and_third, &empty).expect("emits").source;
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

        let key_ascending = kernel_cache_key(&ascending, &empty).expect("cache key builds");
        let key_descending = kernel_cache_key(&descending, &empty).expect("cache key builds");
        assert_ne!(
            key_ascending, key_descending,
            "the SEQUENCE order of output_axes selects which u.output_extents slot each dim reads"
        );

        let source_ascending = emit(&ascending, &empty).expect("emits").source;
        let source_descending = emit(&descending, &empty).expect("emits").source;
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

        let key_q4k = kernel_cache_key(&bound, &q4k).expect("cache key builds");
        let key_q6k = kernel_cache_key(&bound, &q6k).expect("cache key builds");
        assert_ne!(
            key_q4k, key_q6k,
            "entry_name alone cannot see which codec an operand reads through"
        );

        let source_q4k = emit(&bound, &q4k).expect("emits").source;
        let source_q6k = emit(&bound, &q6k).expect("emits").source;
        assert_ne!(source_q4k, source_q6k, "Q4_K and Q6_K unpack through different MSL functions");
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
        let f32_bound = bind(&program, &shapes, &[])
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
        let f16_bound = bind(&half_program, &half_shapes, &[])
            .expect("f16 elementwise lowers")
            .into_iter()
            .next()
            .expect("one bound emitted");

        let empty = BTreeMap::new();
        let key_f32 = kernel_cache_key(&f32_bound, &empty).expect("cache key builds");
        let key_f16 = kernel_cache_key(&f16_bound, &empty).expect("cache key builds");
        assert_ne!(key_f32, key_f16, "entry_name does not encode dtype on its own");

        let source_f32 = emit(&f32_bound, &empty).expect("emits").source;
        let source_f16 = emit(&f16_bound, &empty).expect("emits").source;
        assert_ne!(source_f32, source_f16, "float vs half declarations must differ in source");
    }

    #[test]
    fn same_structure_different_extents_share_one_cache_key() {
        let small = elementwise_tanh_op(4);
        let large = elementwise_tanh_op(4096);
        let empty = BTreeMap::new();

        assert_eq!(
            kernel_cache_key(&small, &empty).expect("cache key builds"),
            kernel_cache_key(&large, &empty).expect("cache key builds"),
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
        let kernel = emit(&bound, &BTreeMap::new()).expect("matmul emits");

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

    #[test]
    fn cumsum_op_emits_a_scan_kernel_with_one_thread_per_line() {
        let bound = cumsum_op(8);
        let kernel = emit(&bound, &BTreeMap::new()).expect("cumsum emits");

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
        let first = emit(&bound, &BTreeMap::new()).expect("first emit succeeds");
        let second = emit(&bound, &BTreeMap::new()).expect("second emit succeeds");
        assert_eq!(first, second);
    }

    #[test]
    fn same_structure_different_extents_yield_identical_source_but_different_grid() {
        let small = elementwise_tanh_op(4);
        let large = elementwise_tanh_op(4096);

        let small_kernel = emit(&small, &BTreeMap::new()).expect("small emits");
        let large_kernel = emit(&large, &BTreeMap::new()).expect("large emits");

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

        let error = emit(&bound, &BTreeMap::new()).expect_err("mismatched arity is rejected");
        assert!(matches!(error, EmitError::ArityMismatch { .. }), "{error}");
    }

    #[test]
    fn a_select_reduction_body_is_rejected() {
        let mut bound = matmul_op(4, 3, 5);
        if let BoundOpKind::Reduce { reduce_op, .. } = &mut bound.kind {
            *reduce_op = ScalarOp::Select;
        }

        let error = emit(&bound, &BTreeMap::new()).expect_err("select reduction body is rejected");
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

        let error = emit(&bound, &BTreeMap::new()).expect_err("an empty scan is rejected");
        assert!(matches!(error, EmitError::EmptyScan { .. }), "{error}");
    }
}
