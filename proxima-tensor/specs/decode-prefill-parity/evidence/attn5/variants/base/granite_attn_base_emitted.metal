#include <metal_stdlib>
using namespace metal;

inline float proxima_erf(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float magnitude = fabs(x);
    float t = 1.0f / fma(0.3275911f, magnitude, 1.0f);
    float poly = t * fma(fma(fma(fma(1.061405429f, t, -1.453152027f), t, 1.421413741f), t, -0.284496736f), t, 0.254829592f);
    return sign * fma(poly, -exp(-magnitude * magnitude), 1.0f);
}


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


// element `index` (0..32) of one Q8_0 block, byte-for-byte the value
// proxima_gguf::quant::q8_0::dequantize_block writes at the same index.
static inline float q8_0_element(device const uchar *block, uint index) {
    ushort d_bits = (ushort)((uint)block[0] | ((uint)block[1] << 8));
    float d = (float)as_type<half>(d_bits);
    char level = (char)block[2u + index];
    return (float)level * d;
}


static inline float q8_0_super_element(device const uchar *superblock, uint index) {
    return q8_0_element(superblock + (index / 32u) * 34u, index % 32u);
}


static inline float q8_0_pair_dot(device const uchar *superblock, uint slot, thread const float *acts) {
    device const uchar *block = superblock + (slot / 32u) * 34u;
    ushort d_bits = (ushort)((uint)block[0] | ((uint)block[1] << 8));
    float d = (float)as_type<half>(d_bits);
    device const uchar *levels = block + 2;
    float result = 0.0f;
    for (uint j = 0u; j < 32u; ++j) {
        result += d * (float)((char)levels[j]) * acts[j];
    }
    return result;
}


// element `index` (0..32) of one Q4_0 block, byte-for-byte the value
// proxima_gguf::quant::q4_0::dequantize_block writes at the same index.
static inline float q4_0_element(device const uchar *block, uint index) {
    ushort d_bits = (ushort)((uint)block[0] | ((uint)block[1] << 8));
    float d = (float)as_type<half>(d_bits);
    uchar byte = block[2u + (index % 16u)];
    int nibble = (index < 16u) ? (int)(byte & 0x0Fu) : (int)(byte >> 4u);
    return (float)(nibble - 8) * d;
}


static inline float q4_0_super_element(device const uchar *superblock, uint index) {
    return q4_0_element(superblock + (index / 32u) * 18u, index % 32u);
}


static inline float q4_0_pair_dot(device const uchar *superblock, uint slot, thread const float *acts) {
    device const uchar *block = superblock + (slot / 32u) * 18u;
    ushort d_bits = (ushort)((uint)block[0] | ((uint)block[1] << 8));
    float d = (float)as_type<half>(d_bits);
    device const uchar *qs = block + 2;
    float result = 0.0f;
    for (uint j = 0u; j < 16u; ++j) {
        uchar byte = qs[j];
        float low = (float)((int)(byte & 0x0Fu) - 8);
        float high = (float)((int)(byte >> 4u) - 8);
        result += d * low * acts[j];
        result += d * high * acts[j + 16u];
    }
    return result;
}


static inline float q4_0_block_scale(device const uchar *block) {
    ushort d_bits = (ushort)((uint)block[0] | ((uint)block[1] << 8));
    return (float)as_type<half>(d_bits);
}

static inline void q4_0_run8(device const uchar *block, uint index, thread float *out) {
    device const uchar *qs = block + 2;
    uint shift = (index < 16u) ? 0u : 4u;
    uint base = index % 16u;
    for (uint j = 0u; j < 8u; ++j) {
        uchar byte = qs[base + j];
        out[j] = (float)((byte >> shift) & 0x0Fu);
    }
}


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


// element `index` (always 0 -- BFloat16 has no super-block) of one BFloat16
// block: the top 16 bits of the f32 this pair of bytes came from,
// reconstructed by shifting them back into place.
static inline float bf16_element(device const uchar *block, uint index) {
    (void)index;
    uint bits = ((uint)block[0] | ((uint)block[1] << 8)) << 16u;
    return as_type<float>(bits);
}

struct Uniforms { long total_elements; long splits; };

#define FOR_UNROLL _Pragma("clang loop unroll(full)")

kernel void omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n2_b64_rt(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const float* in2 [[buffer(2)]], device const float* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const float* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], uint tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort simdgroup_slot [[simdgroup_index_in_threadgroup]]) {
    constexpr long kv_heads = 8; constexpr long query_groups = 2; constexpr long head_dim = 64; constexpr long half_dim = head_dim / 2; constexpr float scale = 0.015625; constexpr long cached_lower = -9223372036854775807L; constexpr long new_upper = 0L;
    constexpr long tile_rows = 8; constexpr long simdgroups = 2; constexpr long block = 64; constexpr long split_keys = 64;
    constexpr bool rows_in_fragment = (query_groups % 8) != 0;
    constexpr long groups_per_row = rows_in_fragment ? 1 : query_groups / 8;
    constexpr long tile_blocks = rows_in_fragment ? (tile_rows / 8) * query_groups : (tile_rows * query_groups) / 8;
    constexpr long tile_vectors = tile_blocks * 8; constexpr long threads = simdgroups * 32;
    constexpr long dims_per_group = head_dim / 8 / simdgroups; constexpr long depth_unroll = ((half_dim / 8) % 2 == 0) ? 2 : 1;
    constexpr long key_tiles_per_group = (block / 8) / simdgroups;
    constexpr long query_stride = rows_in_fragment ? kv_heads * query_groups * half_dim : half_dim;
    long splits = u.splits;
    long total_rows = u.total_elements / (kv_heads * query_groups);
    long tiles = (total_rows + tile_rows - 1L) / tile_rows;
    if ((long)tgid >= kv_heads * tiles * splits) { return; }
    long split = (long)tgid % splits;
    long tile = tiles - 1L - ((long)tgid / splits) % tiles;
    long kv_head = (long)tgid / (splits * tiles);
    long row0 = tile * tile_rows;
    long rows_here = min(tile_rows, total_rows - row0);
    long live = (long)in8[0];
    long thread_id = (long)simdgroup_slot * 32L + (long)lane;
    short quad = (short)(lane / 4); short fragment_row = (short)((quad & 4) + ((lane / 2) % 4)); short fragment_column = (short)((quad & 2) * 2 + (lane % 2) * 2);
    threadgroup float score_tile[tile_vectors * block]; threadgroup float row_maximum[tile_vectors]; threadgroup float row_sum[tile_vectors]; threadgroup float rescale_tile[tile_vectors];
    threadgroup int vector_row[tile_vectors]; threadgroup int vector_head[tile_vectors]; threadgroup int vector_live[tile_vectors];
    constexpr bool stage_query = true; constexpr long query_stage_stride = half_dim + 8L;
    threadgroup float query_stage[stage_query ? tile_blocks * 2L * 8L * query_stage_stride : 1L];
    for (long index = thread_id; index < tile_vectors; index += threads) {
        long block_index = index / 8L; long within = index % 8L;
        long row; long head; long owned;
        if (rows_in_fragment) {
            long row_block = block_index / query_groups;
            long owned_from = row0 + row_block * 8L;
            row = min(owned_from, total_rows - 8L) + within; head = block_index % query_groups;
            owned = (row >= owned_from && row < total_rows) ? 1L : 0L;
        } else {
            row = row0 + index / query_groups; head = index % query_groups;
            owned = (row < total_rows) ? 1L : 0L;
        }
        vector_row[index] = (int)row; vector_head[index] = (int)head; vector_live[index] = (int)owned;
        row_maximum[index] = -INFINITY; row_sum[index] = 0.0f; rescale_tile[index] = 0.0f;
    }
    long block_row[tile_blocks]; long block_head[tile_blocks];
    FOR_UNROLL for (long vector_block = 0L; vector_block < tile_blocks; vector_block++) {
        if (rows_in_fragment) {
            block_row[vector_block] = min(row0 + (vector_block / query_groups) * 8L, total_rows - 8L); block_head[vector_block] = vector_block % query_groups;
        } else {
            block_row[vector_block] = min(row0 + vector_block / groups_per_row, total_rows - 1L); block_head[vector_block] = (vector_block % groups_per_row) * 8L;
        }
    }
    simdgroup_float8x8 accumulated[dims_per_group][tile_blocks];
    FOR_UNROLL for (long slot = 0L; slot < dims_per_group; slot++) { FOR_UNROLL for (long vector_block = 0L; vector_block < tile_blocks; vector_block++) { accumulated[slot][vector_block] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); } }
    if (stage_query) {
        FOR_UNROLL for (long vector_block = 0L; vector_block < tile_blocks; vector_block++) {
            long query_base = (block_row[vector_block] * (kv_heads * query_groups) + kv_head * query_groups + block_head[vector_block]) * half_dim;
            for (long index = thread_id; index < 8L * half_dim; index += threads) {
                long stage_row = index / half_dim; long stage_column = index % half_dim;
                long source = query_base + stage_row * query_stride + stage_column;
                query_stage[(vector_block * 16L + stage_row) * query_stage_stride + stage_column] = in0[source];
                query_stage[(vector_block * 16L + 8L + stage_row) * query_stage_stride + stage_column] = in1[source];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long last_row = row0 + rows_here - 1L;
    long first_key = max(0L, live + cached_lower + row0) & ~7L;
    long band = max(0L, live - first_key);
    long slice = ((((band + splits - 1L) / splits) + split_keys - 1L) / split_keys) * split_keys;
    long slice_start = first_key + split * slice;
    long slice_end = min(slice_start + slice, live);
    long cached_blocks = slice_start < slice_end ? (slice_end - slice_start + block - 1L) / block : 0L;
    long total_aligned = total_rows & ~7L;
    long new_first = max(0L, row0 + cached_lower);
    long new_end = min(total_rows, last_row + new_upper + 1L);
    long new_start = new_first & ~7L;
    long mma_end = min(new_end, total_aligned);
    bool last_split = split == splits - 1L;
    long new_blocks = (last_split && mma_end > new_start) ? (mma_end - new_start + block - 1L) / block : 0L;
    long tail_steps = (last_split && new_end > total_aligned && new_first < total_rows) ? 1L : 0L;
    for (long step = 0L; step < cached_blocks + new_blocks + tail_steps; step++) {
        long mode = step < cached_blocks ? 0L : (step < cached_blocks + new_blocks ? 1L : 2L);
        long key0 = mode == 0L ? slice_start + step * block : (mode == 1L ? new_start + (step - cached_blocks) * block : total_aligned);
        long columns = mode == 0L ? min(block, slice_end - key0) : (mode == 1L ? min(block, mma_end - key0) : total_rows - total_aligned);
        if (mode != 2L) {
            device const float* key_even = mode == 0L ? in2 : in4;
            device const float* key_odd = mode == 0L ? in3 : in5;
            int fragments = (int)((columns + 7L) / 8L);
            simdgroup_float8x8 scores[key_tiles_per_group][tile_blocks];
            FOR_UNROLL for (int group = 0; group < (int)key_tiles_per_group; group++) { FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) { scores[group][vector_block] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); } }
            for (int depth = 0; depth < (int)half_dim; depth += 8 * (int)depth_unroll) {
                simdgroup_float8x8 key_even_tile[key_tiles_per_group][depth_unroll]; simdgroup_float8x8 key_odd_tile[key_tiles_per_group][depth_unroll];
                FOR_UNROLL for (int group = 0; group < (int)key_tiles_per_group; group++) {
                    int key_tile = (int)simdgroup_slot + group * (int)simdgroups;
                    if (key_tile < fragments) {
                        device const float* key_even_ptr = key_even + (key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim;
                        device const float* key_odd_ptr = key_odd + (key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim;
                        FOR_UNROLL for (int step_index = 0; step_index < (int)depth_unroll; step_index++) {
                            simdgroup_load(key_even_tile[group][step_index], key_even_ptr + depth + 8 * step_index, (ulong)(kv_heads * half_dim), ulong2(0, 0), true);
                            simdgroup_load(key_odd_tile[group][step_index], key_odd_ptr + depth + 8 * step_index, (ulong)(kv_heads * half_dim), ulong2(0, 0), true);
                        }
                    }
                }
                FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) {
                    long query_offset = (block_row[vector_block] * (kv_heads * query_groups) + kv_head * query_groups + block_head[vector_block]) * half_dim + depth;
                    simdgroup_float8x8 query_even[depth_unroll]; simdgroup_float8x8 query_odd[depth_unroll];
                    FOR_UNROLL for (int step_index = 0; step_index < (int)depth_unroll; step_index++) {
                        if (stage_query) {
                            threadgroup const float* stage_even = query_stage + (long)vector_block * 16L * query_stage_stride + depth + 8 * step_index;
                            simdgroup_load(query_even[step_index], stage_even, (ulong)query_stage_stride);
                            simdgroup_load(query_odd[step_index], stage_even + 8L * query_stage_stride, (ulong)query_stage_stride);
                        } else {
                            simdgroup_load(query_even[step_index], in0 + query_offset + 8 * step_index, (ulong)query_stride);
                            simdgroup_load(query_odd[step_index], in1 + query_offset + 8 * step_index, (ulong)query_stride);
                        }
                    }
                    FOR_UNROLL for (int group = 0; group < (int)key_tiles_per_group; group++) {
                        if ((int)simdgroup_slot + group * (int)simdgroups < fragments) {
                            FOR_UNROLL for (int step_index = 0; step_index < (int)depth_unroll; step_index++) {
                                simdgroup_multiply_accumulate(scores[group][vector_block], query_even[step_index], key_even_tile[group][step_index], scores[group][vector_block]);
                                simdgroup_multiply_accumulate(scores[group][vector_block], query_odd[step_index], key_odd_tile[group][step_index], scores[group][vector_block]);
                            }
                        }
                    }
                }
            }
            FOR_UNROLL for (int group = 0; group < (int)key_tiles_per_group; group++) {
                int key_tile = (int)simdgroup_slot + group * (int)simdgroups;
                if (key_tile < fragments) {
                    FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) { simdgroup_store(scores[group][vector_block], score_tile + vector_block * 8 * (int)block + key_tile * 8, (ulong)block); }
                }
            }
        } else {
            long pairs = tile_vectors * columns;
            short ty = (short)(lane / 8); short tx = (short)(lane % 8);
            for (long group = (long)simdgroup_slot; group * 4L < pairs; group += simdgroups) {
                long pair = group * 4L + (long)ty;
                long vector = pair / columns;
                long column = pair % columns;
                long query_row = (long)vector_row[vector];
                long relative = key0 + column - query_row;
                bool valid = pair < pairs && vector_live[vector] != 0 && relative <= new_upper && relative >= cached_lower;
                float partial_score = 0.0f;
                if (valid) {
                    long query_index = query_row * (kv_heads * query_groups) + kv_head * query_groups + (long)vector_head[vector];
                    long key_offset = (key0 + column) * (kv_heads * half_dim) + kv_head * half_dim;
                    device const float4* query_even4 = (device const float4*)(in0 + query_index * half_dim);
                    device const float4* query_odd4 = (device const float4*)(in1 + query_index * half_dim);
                    device const float4* key_even4 = (device const float4*)(in4 + key_offset);
                    device const float4* key_odd4 = (device const float4*)(in5 + key_offset);
                    for (short index = tx; index < (short)(half_dim / 4L); index += 8) {
                        partial_score += dot(key_even4[index], query_even4[index]);
                        partial_score += dot(key_odd4[index], query_odd4[index]);
                    }
                }
                partial_score += simd_shuffle_down(partial_score, 4);
                partial_score += simd_shuffle_down(partial_score, 2);
                partial_score += simd_shuffle_down(partial_score, 1);
                if (tx == 0 && pair < pairs) { score_tile[vector * block + column] = valid ? partial_score * scale : -INFINITY; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (long vector = (long)simdgroup_slot; vector < tile_vectors; vector += simdgroups) {
            long query_row = (long)vector_row[vector];
            float local_scores[block / 32];
            float block_maximum = -INFINITY;
            FOR_UNROLL for (long item = 0L; item < block / 32L; item++) {
                long column = (long)lane + item * 32L;
                float raw_score = -INFINITY;
                if (item * 32L < columns) {
                    if (mode == 2L) {
                        if (column < columns) { raw_score = score_tile[vector * block + column]; }
                    } else if (mode == 1L) {
                        long relative = key0 + column - query_row;
                        if (column < columns && relative <= new_upper && relative >= cached_lower) { raw_score = score_tile[vector * block + column] * scale; }
                    } else {
                        long key = key0 + column;
                        if (key < slice_end && (key - live - query_row) >= cached_lower) { raw_score = score_tile[vector * block + column] * scale; }
                    }
                }
                local_scores[item] = raw_score;
                block_maximum = max(block_maximum, raw_score);
            }
            block_maximum = simd_max(block_maximum);
            float previous_maximum = row_maximum[vector];
            float next_maximum = max(previous_maximum, block_maximum);
            float rescale = (previous_maximum == -INFINITY) ? 0.0f : exp(previous_maximum - next_maximum);
            float block_sum = 0.0f;
            FOR_UNROLL for (long item = 0L; item < block / 32L; item++) {
                if (item * 32L < columns) {
                    float weight = (local_scores[item] == -INFINITY) ? 0.0f : exp(local_scores[item] - next_maximum);
                    score_tile[vector * block + (long)lane + item * 32L] = weight;
                    block_sum += weight;
                }
            }
            block_sum = simd_sum(block_sum);
            if (lane == 0) { row_maximum[vector] = next_maximum; row_sum[vector] = row_sum[vector] * rescale + block_sum; rescale_tile[vector] = rescale; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) {
            float row_scale = rescale_tile[vector_block * 8 + fragment_row];
            FOR_UNROLL for (int slot = 0; slot < (int)dims_per_group; slot++) {
                accumulated[slot][vector_block].thread_elements()[0] *= row_scale;
                accumulated[slot][vector_block].thread_elements()[1] *= row_scale;
            }
        }
        if (mode != 2L) {
            device const float* value_ptr = (mode == 0L ? in6 : in7) + key0 * (kv_heads * head_dim) + kv_head * head_dim;
            int fragments = (int)((columns + 7L) / 8L);
            for (int key_tile = 0; key_tile < fragments; key_tile++) {
                simdgroup_float8x8 weights[tile_blocks];
                FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) { simdgroup_load(weights[vector_block], score_tile + vector_block * 8 * (int)block + key_tile * 8, (ulong)block); }
                FOR_UNROLL for (int slot = 0; slot < (int)dims_per_group; slot++) {
                    int dimension_block = (int)simdgroup_slot + slot * (int)simdgroups;
                    simdgroup_float8x8 value;
                    simdgroup_load(value, value_ptr + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8, (ulong)(kv_heads * head_dim));
                    FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) { simdgroup_multiply_accumulate(accumulated[slot][vector_block], weights[vector_block], value, accumulated[slot][vector_block]); }
                }
            }
        } else {
            FOR_UNROLL for (int slot = 0; slot < (int)dims_per_group; slot++) {
                long dimension = ((long)simdgroup_slot + (long)slot * simdgroups) * 8L + (long)fragment_column;
                FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) {
                    float sum_even = 0.0f; float sum_odd = 0.0f;
                    for (long column = 0L; column < columns; column++) {
                        float weight = score_tile[(vector_block * 8 + fragment_row) * block + column];
                        long value_offset = (key0 + column) * (kv_heads * head_dim) + kv_head * head_dim + dimension;
                        sum_even += weight * in7[value_offset];
                        sum_odd += weight * in7[value_offset + 1L];
                    }
                    accumulated[slot][vector_block].thread_elements()[0] += sum_even;
                    accumulated[slot][vector_block].thread_elements()[1] += sum_odd;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    FOR_UNROLL for (int slot = 0; slot < (int)dims_per_group; slot++) {
        long dimension = ((long)simdgroup_slot + (long)slot * simdgroups) * 8L + (long)fragment_column;
        FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) {
            long vector = (long)vector_block * 8L + (long)fragment_row;
            if (vector_live[vector] != 0) {
                long query_index = (long)vector_row[vector] * (kv_heads * query_groups) + kv_head * query_groups + (long)vector_head[vector];
                if (splits == 1L) {
                    float sum = row_sum[vector];
                    float inverse = sum == 0.0f ? 0.0f : 1.0f / sum;
                    out[query_index * head_dim + dimension] = accumulated[slot][vector_block].thread_elements()[0] * inverse;
                    out[query_index * head_dim + dimension + 1L] = accumulated[slot][vector_block].thread_elements()[1] * inverse;
                } else {
                    device float* attn_scratch = out;
                    long base = ((query_index * (head_dim / 4L) + (dimension >> 2)) * splits + split) * 4L + (dimension & 3L);
                    attn_scratch[base] = accumulated[slot][vector_block].thread_elements()[0];
                    attn_scratch[base + 1L] = accumulated[slot][vector_block].thread_elements()[1];
                }
            }
        }
    }
    if (splits > 1L) {
        device float* attn_scratch = out;
        for (long vector = thread_id; vector < tile_vectors; vector += threads) {
            if (vector_live[vector] != 0) {
                long query_index = (long)vector_row[vector] * (kv_heads * query_groups) + kv_head * query_groups + (long)vector_head[vector];
                long stats_index = u.total_elements * head_dim * splits + (query_index * splits + split) * 2L;
                attn_scratch[stats_index] = row_maximum[vector];
                attn_scratch[stats_index + 1L] = row_sum[vector];
            }
        }
    }
}
