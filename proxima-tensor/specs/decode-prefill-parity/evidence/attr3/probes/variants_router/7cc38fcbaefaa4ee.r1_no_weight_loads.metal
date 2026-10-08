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

struct Uniforms {
    long output_total;
    long reduction_total;
    long output_extents[2];
    long reduction_extents[1];
    long operand_base[2];
    long operand_strides[2][3];
    long out_base;
    long out_strides[3];
};

kernel void omega_reduce_r3_o2_n2_multiply_add_zero(
    device const float* in0 [[buffer(0)]],
    device const float* in1 [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant Uniforms& u [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    long dense_batch_index = (long)tgpig.z;
    long dense_weight_batch_base = 0;
    long dense_other_batch_base = 0;
    long feature_extent = u.output_extents[1];
    long token_extent = u.output_extents[0];
    long row_tile = (long)tgpig.y;
    long col_tile = (long)tgpig.x;
    long row_half = sgitg & 1;
    long col_half = sgitg >> 1;
    threadgroup uchar tg_shared[12288];
    threadgroup float *weight_tile = (threadgroup float *)tg_shared;
    threadgroup float *act_tile = (threadgroup float *)(tg_shared + 8192);
    simdgroup_float8x8 acc[8];
    for (int i = 0; i < 8; ++i) { acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f); }
    for (int k0 = 0; k0 < u.reduction_total; k0 += 32) {
        for (long w_row = tiitg; w_row < 64; w_row += 128) {
            long w_feat = row_tile * 64 + w_row;
            if (w_feat < feature_extent) {
                long row_base = u.operand_base[0] + dense_weight_batch_base + w_feat * u.operand_strides[0][2] + k0 * u.operand_strides[0][1];
                for (long w_k = 0; w_k < 32; ++w_k) {
                    if (k0 + w_k < u.reduction_total) {
                        long w_off = row_base + w_k * u.operand_strides[0][1];
                        weight_tile[w_row * 32 + w_k] = 1.0f;
                    } else {
                        weight_tile[w_row * 32 + w_k] = 0.0f;
                    }
                }
            } else {
                for (long fill_k = 0; fill_k < 32; ++fill_k) { weight_tile[w_row * 32 + fill_k] = 0.0f; }
            }
        }
        for (long idx = tiitg; idx < 1024; idx += 128) {
            long a_col = idx / 32;
            long a_k = idx % 32;
            long a_tok = col_tile * 32 + a_col;
            long a_k_global = k0 + a_k;
            float a_value = 0.0f;
            if (a_tok < token_extent && a_k_global < u.reduction_total) {
                long aoff = u.operand_base[1] + dense_other_batch_base + a_tok * u.operand_strides[1][0] + a_k_global * u.operand_strides[1][1];
                a_value = in1[aoff];
            }
            act_tile[a_col * 32 + a_k] = a_value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int sub_k = 0; sub_k < 4; ++sub_k) {
            simdgroup_float8x8 a_frag[4];
            for (int i = 0; i < 4; ++i) {
                simdgroup_load(a_frag[i], weight_tile + (row_half * 4 + i) * 8 * 32 + sub_k * 8, 32);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_float8x8 b_frag[2];
            for (int j = 0; j < 2; ++j) {
                simdgroup_load(b_frag[j], act_tile + (col_half * 2 + j) * 8 * 32 + sub_k * 8, 32, ulong2(0), true);
            }
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 2; ++j) {
                    simdgroup_multiply_accumulate(acc[i * 2 + j], a_frag[i], b_frag[j], acc[i * 2 + j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float *out_tile = (threadgroup float *)tg_shared;
    bool direct_store_interior = (row_tile * 64 + 64 <= feature_extent) && (col_tile * 32 + 32 <= token_extent) && (u.out_strides[2] == 1);
    if (direct_store_interior) {
        long direct_coord[3];
        for (int d = 0; d < 3; ++d) { direct_coord[d] = 0; }
        direct_coord[2] = row_tile * 64;
        direct_coord[0] = col_tile * 32;
        long direct_offset = u.out_base;
        direct_offset += direct_coord[0] * u.out_strides[0];
        direct_offset += direct_coord[1] * u.out_strides[1];
        direct_offset += direct_coord[2] * u.out_strides[2];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 2; ++j) {
                simdgroup_store(acc[i * 2 + j], out + direct_offset + (row_half * 4 + i) * 8 * u.out_strides[2] + (col_half * 2 + j) * 8 * u.out_strides[0], u.out_strides[0], ulong2(0), true);
            }
        }
    } else {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            simdgroup_store(acc[i * 2 + j], out_tile + (row_half * 4 + i) * 8 * 32 + (col_half * 2 + j) * 8, 32);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long coord[3];
    for (int d = 0; d < 3; ++d) { coord[d] = 0; }
    long out_offset_base = u.out_base;
    for (long idx = tiitg; idx < 2048; idx += 128) {
        long o_row = idx / 32;
        long o_col = idx % 32;
        long o_feat = row_tile * 64 + o_row;
        long o_tok = col_tile * 32 + o_col;
        if (o_feat < feature_extent && o_tok < token_extent) {
            coord[2] = o_feat;
            coord[0] = o_tok;
            long out_offset = out_offset_base + o_feat * u.out_strides[2] + o_tok * u.out_strides[0];
            out[out_offset] = (float)out_tile[idx];
        }
    }
    }
}

