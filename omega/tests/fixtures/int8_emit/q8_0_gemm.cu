#include <cuda_fp16.h>
__device__ __forceinline__ void omega_q8_1_quantize_block(const float *in, float *out, long group, unsigned int lane) {
    float x = in[group * 32 + lane];
    float amax = fabsf(x);
    for (int offset = 16; offset > 0; offset >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset)); }
    float d = amax / 127.0f;
    int q = amax == 0.0f ? 0 : (int)roundf(x / d);
    int qsum = q;
    for (int offset = 16; offset > 0; offset >>= 1) { qsum += __shfl_xor_sync(0xffffffffu, qsum, offset); }
    unsigned char *bytes = (unsigned char *)out + (group / 4) * 144;
    const long run = group % 4;
    unsigned char *header = bytes + 0 + run * 4;
    if (lane == 0) {
        unsigned short d_bits = __half_as_ushort(__float2half(d));
        unsigned short s_bits = __half_as_ushort(__float2half(d * (float)qsum));
        header[0] = (unsigned char)(d_bits & 0xffu);
        header[1] = (unsigned char)(d_bits >> 8);
        header[2] = (unsigned char)(s_bits & 0xffu);
        header[3] = (unsigned char)(s_bits >> 8);
    }
    bytes[16 + run * 32 + lane] = (unsigned char)(signed char)q;
}

struct Uniforms { long groups; };

extern "C" __global__ void __launch_bounds__(256) omega_cuda_block_quantize_e128_b144_wide_gid64c8(
    const float* __restrict__ in0,
    float* __restrict__ out,
    const Uniforms* __restrict__ u_ptr
) {
    const Uniforms u = *u_ptr;
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long group = gid / 32;
    if (group >= u.groups) { return; }
    omega_q8_1_quantize_block(in0, out, group, (unsigned int)(gid & 31));
}
#include <cuda_fp16.h>

__device__ __forceinline__ float proxima_erf(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float magnitude = fabsf(x);
    float t = 1.0f / fmaf(0.3275911f, magnitude, 1.0f);
    float poly = t * fmaf(fmaf(fmaf(fmaf(1.061405429f, t, -1.453152027f), t, 1.421413741f), t, -0.284496736f), t, 0.254829592f);
    return sign * fmaf(poly, -expf(-magnitude * magnitude), 1.0f);
}


__device__ __forceinline__ float q2k_element(const unsigned char *block, unsigned int index) {
    const unsigned char *scales = block;
    const unsigned char *qs = block + 16;
    unsigned short d_bits = (unsigned short)((unsigned int)block[80] | ((unsigned int)block[81] << 8));
    unsigned short dmin_bits = (unsigned short)((unsigned int)block[82] | ((unsigned int)block[83] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    float dmin = __half2float(__ushort_as_half(dmin_bits));
    unsigned int chunk = index / 128u;
    unsigned int within = index % 128u;
    unsigned int group = within / 32u;
    unsigned int local = within % 32u;
    unsigned int sub_block = chunk * 8u + group * 2u + (local >= 16u ? 1u : 0u);
    unsigned char scale_min = scales[sub_block];
    float scale = d * (float)(scale_min & 0x0Fu);
    float minimum = dmin * (float)(scale_min >> 4u);
    unsigned char level = (qs[chunk * 32u + local] >> (2u * group)) & 0x03u;
    return scale * (float)level - minimum;
}


__device__ __forceinline__ int q3k_unpack_scale(const unsigned char *scales, unsigned int sub_block) {
    unsigned char low = (sub_block < 8u) ? (scales[sub_block] & 0x0Fu) : (scales[sub_block - 8u] >> 4u);
    unsigned char high = (scales[8u + sub_block % 4u] >> (2u * (sub_block / 4u))) & 0x03u;
    return (int)(low | (high << 4u)) - 32;
}
struct q3k_header_cuda { float scale; unsigned char mask; };
__device__ __forceinline__ q3k_header_cuda q3k_header_for(const unsigned char *block, unsigned int index) {
    const unsigned char *scales = block + 96;
    unsigned short d_bits = (unsigned short)((unsigned int)block[108] | ((unsigned int)block[109] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned int chunk = index / 128u;
    unsigned int rem = index % 128u;
    unsigned int j = rem / 32u;
    unsigned int local32 = rem % 32u;
    q3k_header_cuda header;
    header.scale = d * (float)q3k_unpack_scale(scales, 8u * chunk + 2u * j + (local32 < 16u ? 0u : 1u));
    header.mask = (unsigned char)(1u << (4u * chunk + j));
    return header;
}
__device__ __forceinline__ float q3k_element(const unsigned char *block, unsigned int index) {
    q3k_header_cuda header = q3k_header_for(block, index);
    const unsigned char *qs = block + 32;
    unsigned int chunk = index / 128u;
    unsigned int rem = index % 128u;
    unsigned int j = rem / 32u;
    unsigned int local32 = rem % 32u;
    unsigned char level = (qs[chunk * 32u + local32] >> (2u * j)) & 0x03u;
    float correction = (block[local32] & header.mask) != 0u ? 0.0f : 4.0f;
    return header.scale * ((float)level - correction);
}

__device__ __forceinline__ unsigned char q4k_scale(const unsigned char *scales, unsigned int sub_block) {
    if (sub_block < 4u) { return scales[sub_block] & 63; }
    return (scales[sub_block + 4u] & 0x0F) | ((scales[sub_block - 4u] >> 6) << 4);
}

__device__ __forceinline__ unsigned char q4k_min(const unsigned char *scales, unsigned int sub_block) {
    if (sub_block < 4u) { return scales[sub_block + 4u] & 63; }
    return (scales[sub_block + 4u] >> 4) | ((scales[sub_block] >> 6) << 4);
}

__device__ __forceinline__ float q4k_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    unsigned short dmin_bits = (unsigned short)((unsigned int)block[2] | ((unsigned int)block[3] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    float dmin = __half2float(__ushort_as_half(dmin_bits));

    const unsigned char *scales = block + 4;
    const unsigned char *qs = block + 16;

    unsigned int group = index / 64u;
    unsigned int within = index % 64u;
    int low_nibble = within < 32u;
    unsigned int sub_block = 2u * group + (low_nibble ? 0u : 1u);
    unsigned int byte_index = group * 32u + (within % 32u);

    float scale = d * (float)q4k_scale(scales, sub_block);
    float minimum = dmin * (float)q4k_min(scales, sub_block);
    unsigned char nibble = low_nibble ? (qs[byte_index] & 0x0Fu) : (qs[byte_index] >> 4u);
    return scale * (float)nibble - minimum;
}

__device__ __forceinline__ void q4k_chunk8(const unsigned char *block, unsigned int index, float *out) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    unsigned short dmin_bits = (unsigned short)((unsigned int)block[2] | ((unsigned int)block[3] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    float dmin = __half2float(__ushort_as_half(dmin_bits));

    const unsigned char *scales = block + 4;
    unsigned int group = index / 64u;
    unsigned int within = index % 64u;
    int low_nibble = within < 32u;
    unsigned int sub_block = 2u * group + (low_nibble ? 0u : 1u);
    unsigned int byte_index = group * 32u + (within % 32u);

    float scale = d * (float)q4k_scale(scales, sub_block);
    float minimum = dmin * (float)q4k_min(scales, sub_block);
    uint2 packed = *(const uint2 *)(block + 16u + byte_index);
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        unsigned int word = slot < 4 ? packed.x : packed.y;
        unsigned int byte = (word >> (8u * (slot & 3))) & 0xFFu;
        unsigned int nibble = low_nibble ? (byte & 0x0Fu) : (byte >> 4u);
        out[slot] = scale * (float)nibble - minimum;
    }
}

__device__ __forceinline__ float q5k_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    unsigned short dmin_bits = (unsigned short)((unsigned int)block[2] | ((unsigned int)block[3] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    float dmin = __half2float(__ushort_as_half(dmin_bits));
    const unsigned char *scales = block + 4;
    const unsigned char *qh = block + 16;
    const unsigned char *qs = block + 48;

    unsigned int chunk = index / 64u;
    unsigned int within = index % 64u;
    int low = within < 32u;
    unsigned int sub_block = 2u * chunk + (low ? 0u : 1u);
    unsigned int offset = within % 32u;

    unsigned char scale = q4k_scale(scales, sub_block);
    unsigned char minimum = q4k_min(scales, sub_block);
    unsigned char mask = low ? (unsigned char)(1u << (2u * chunk)) : (unsigned char)(2u << (2u * chunk));

    unsigned char qs_byte = qs[chunk * 32u + offset];
    unsigned char nibble = low ? (qs_byte & 0x0Fu) : (qs_byte >> 4u);
    float high = (qh[offset] & mask) != 0u ? 16.0f : 0.0f;
    return d * (float)scale * ((float)nibble + high) - dmin * (float)minimum;
}

__device__ __forceinline__ float q6k_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[208] | ((unsigned int)block[209] << 8));
    float d = __half2float(__ushort_as_half(d_bits));

    unsigned int half_index = index / 128u;
    unsigned int local = index % 128u;
    unsigned int l = local % 32u;
    unsigned int lane = local / 32u;
    unsigned int sub_block_in_half = l / 16u;

    const unsigned char *ql = block + half_index * 64u;
    const unsigned char *qh = block + 128u + half_index * 32u;
    const unsigned char *scales = block + 192u;

    unsigned char ql_byte = (lane % 2u == 0u) ? ql[l] : ql[l + 32u];
    unsigned char nibble = (lane < 2u) ? (ql_byte & 0x0Fu) : (ql_byte >> 4u);
    unsigned char high2 = (qh[l] >> (unsigned char)(lane * 2u)) & 0x03u;
    unsigned char level = nibble | (high2 << 4u);

    unsigned char scale_byte = scales[half_index * 8u + sub_block_in_half + lane * 2u];
    float scale = (float)(signed char)scale_byte;
    float quant = (float)level - 32.0f;
    return d * scale * quant;
}
__device__ __forceinline__ void q6k_chunk8(const unsigned char *block, unsigned int index, float *out) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[208] | ((unsigned int)block[209] << 8));
    float d = __half2float(__ushort_as_half(d_bits));

    unsigned int half_index = index / 128u;
    unsigned int local = index % 128u;
    unsigned int l = local % 32u;
    unsigned int lane = local / 32u;
    unsigned int sub_block_in_half = l / 16u;

    const unsigned short *ql_pairs = (const unsigned short *)(block + half_index * 64u + (lane % 2u) * 32u + l);
    const unsigned short *qh_pairs = (const unsigned short *)(block + 128u + half_index * 32u + l);
    float scale = (float)(signed char)block[192u + half_index * 8u + sub_block_in_half + lane * 2u];
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        unsigned int ql_word = (unsigned int)ql_pairs[pair];
        unsigned int qh_word = (unsigned int)qh_pairs[pair];
        #pragma unroll
        for (int byte = 0; byte < 2; byte++) {
            unsigned int ql_byte = (ql_word >> (8u * byte)) & 0xFFu;
            unsigned int qh_byte = (qh_word >> (8u * byte)) & 0xFFu;
            unsigned int nibble = (lane < 2u) ? (ql_byte & 0x0Fu) : (ql_byte >> 4u);
            unsigned int high2 = (qh_byte >> (lane * 2u)) & 0x03u;
            unsigned int level = nibble | (high2 << 4u);
            out[2 * pair + byte] = d * scale * ((float)level - 32.0f);
        }
    }
}

__device__ __forceinline__ float q8_0_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    signed char level = (signed char)block[2u + index];
    return (float)level * d;
}
__device__ __forceinline__ void q8_0_chunk8(const unsigned char *block, unsigned int index, float *out) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    const unsigned short *pairs = (const unsigned short *)(block + 2u + index);
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        unsigned int word = (unsigned int)pairs[pair];
        out[2 * pair] = (float)((signed char)(word & 0xFFu)) * d;
        out[2 * pair + 1] = (float)((signed char)(word >> 8u)) * d;
    }
}

__device__ __forceinline__ float q4_0_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned char byte = block[2u + (index % 16u)];
    int nibble = (index < 16u) ? (int)(byte & 0x0Fu) : (int)(byte >> 4u);
    return (float)(nibble - 8) * d;
}
__device__ __forceinline__ void q4_0_chunk8(const unsigned char *block, unsigned int index, float *out) {
    const unsigned short *halves = (const unsigned short *)block;
    float d = __half2float(__ushort_as_half(halves[0]));
    const unsigned short *pairs = halves + 1u + (index % 16u) / 2u;
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        unsigned int word = (unsigned int)pairs[pair];
        unsigned int low = word & 0xFFu;
        unsigned int high = word >> 8u;
        int low_nibble = (index < 16u) ? (int)(low & 0x0Fu) : (int)(low >> 4u);
        int high_nibble = (index < 16u) ? (int)(high & 0x0Fu) : (int)(high >> 4u);
        out[2 * pair] = (float)(low_nibble - 8) * d;
        out[2 * pair + 1] = (float)(high_nibble - 8) * d;
    }
}

__device__ __forceinline__ float q5_0_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned int qh = (unsigned int)block[2] | ((unsigned int)block[3] << 8) | ((unsigned int)block[4] << 16) | ((unsigned int)block[5] << 24);
    bool high_half = index >= 16u;
    unsigned int local = high_half ? (index - 16u) : index;
    unsigned int byte = (unsigned int)block[6u + local];
    unsigned int level;
    if (high_half) {
        level = (byte >> 4u) | ((qh >> (local + 12u)) & 0x10u);
    } else {
        level = (byte & 0x0Fu) | (((qh >> local) << 4u) & 0x10u);
    }
    return ((float)level - 16.0f) * d;
}

__device__ __forceinline__ float q1_0_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned int bit = ((unsigned int)block[2u + (index >> 3u)] >> (index & 7u)) & 1u;
    return bit ? d : -d;
}
__device__ __forceinline__ void q1_0_chunk8(const unsigned char *block, unsigned int index, float *out) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned int bits = (unsigned int)block[2u + (index >> 3u)];
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        out[slot] = ((bits >> slot) & 1u) ? d : -d;
    }
}

__device__ __forceinline__ float q2_0_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned int code = ((unsigned int)block[2u + (index >> 2u)] >> ((index & 3u) * 2u)) & 3u;
    return ((float)code - 1.0f) * d;
}
__device__ __forceinline__ void q2_0_chunk8(const unsigned char *block, unsigned int index, float *out) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned int bits = (unsigned int)*(const unsigned short *)(block + 2u + (index >> 2u));
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        out[slot] = ((float)((bits >> (2u * slot)) & 3u) - 1.0f) * d;
    }
}

__device__ __forceinline__ float ptq1_0_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[26] | ((unsigned int)block[27] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned int byte_index;
    unsigned int position;
    if (index < 80u) {
        byte_index = index & 15u;
        position = index >> 4u;
    } else if (index < 120u) {
        byte_index = 16u + ((index - 80u) & 7u);
        position = (index - 80u) >> 3u;
    } else {
        byte_index = 24u + ((index - 120u) & 1u);
        position = (index - 120u) >> 1u;
    }
    unsigned int power = position == 0u ? 1u : position == 1u ? 3u : position == 2u ? 9u : position == 3u ? 27u : 81u;
    unsigned int scaled = ((unsigned int)block[byte_index] * power) & 255u;
    return (float)((int)((scaled * 3u) >> 8u) - 1) * d;
}

__device__ __forceinline__ float bf16_element(const unsigned char *block, unsigned int index) {
    (void)index;
    unsigned int bits = ((unsigned int)block[0] | ((unsigned int)block[1] << 8)) << 16u;
    return __uint_as_float(bits);
}


__device__ __forceinline__ void mma_m16n8k8(float *c, unsigned int a0, unsigned int a1, unsigned int b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ float mma_half_scale(const unsigned char *bytes) {
    return __half2float(__ushort_as_half((unsigned short)((unsigned int)bytes[0] | ((unsigned int)bytes[1] << 8))));
}
__device__ __forceinline__ __half2 mma_pair(int low, int high) {
    return __halves2half2(__int2half_rn(low), __int2half_rn(high));
}
__device__ __forceinline__ void mma_decode_q8_0(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    *scale = mma_half_scale(block);
    *minimum = 0.0f;
    const signed char *quants = (const signed char *)(block + 2u + within);
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) { out[pair] = mma_pair((int)quants[2 * pair], (int)quants[2 * pair + 1]); }
}
__device__ __forceinline__ void mma_decode_q4_0(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    *scale = mma_half_scale(block);
    *minimum = 0.0f;
    const unsigned char *bytes = block + 2u + (within & 15u);
    bool high = within >= 16u;
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        unsigned int low_byte = bytes[2 * pair];
        unsigned int high_byte = bytes[2 * pair + 1];
        int low_q = (int)(high ? (low_byte >> 4u) : (low_byte & 0x0Fu)) - 8;
        int high_q = (int)(high ? (high_byte >> 4u) : (high_byte & 0x0Fu)) - 8;
        out[pair] = mma_pair(low_q, high_q);
    }
}
__device__ __forceinline__ int mma_q5_0_level(const unsigned char *block, unsigned int qh, unsigned int index) {
    bool high_half = index >= 16u;
    unsigned int local = high_half ? (index - 16u) : index;
    unsigned int byte = (unsigned int)block[6u + local];
    return high_half ? (int)((byte >> 4u) | ((qh >> (local + 12u)) & 0x10u)) : (int)((byte & 0x0Fu) | (((qh >> local) << 4u) & 0x10u));
}
__device__ __forceinline__ void mma_decode_q5_0(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    *scale = mma_half_scale(block);
    *minimum = 0.0f;
    unsigned int qh = (unsigned int)block[2] | ((unsigned int)block[3] << 8) | ((unsigned int)block[4] << 16) | ((unsigned int)block[5] << 24);
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        out[pair] = mma_pair(mma_q5_0_level(block, qh, within + 2 * pair) - 16, mma_q5_0_level(block, qh, within + 2 * pair + 1) - 16);
    }
}
__device__ __forceinline__ void mma_decode_q4k(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    float d = mma_half_scale(block);
    float dmin = mma_half_scale(block + 2);
    const unsigned char *scales = block + 4;
    const unsigned char *qs = block + 16;
    unsigned int group = within / 64u;
    unsigned int within_group = within % 64u;
    bool low = within_group < 32u;
    unsigned int sub_block = 2u * group + (low ? 0u : 1u);
    *scale = d * (float)q4k_scale(scales, sub_block);
    *minimum = dmin * (float)q4k_min(scales, sub_block);
    const unsigned char *bytes = qs + group * 32u + (within_group % 32u);
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        unsigned int low_byte = bytes[2 * pair];
        unsigned int high_byte = bytes[2 * pair + 1];
        out[pair] = mma_pair((int)(low ? (low_byte & 0x0Fu) : (low_byte >> 4u)), (int)(low ? (high_byte & 0x0Fu) : (high_byte >> 4u)));
    }
}
__device__ __forceinline__ void mma_decode_q5k(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    float d = mma_half_scale(block);
    float dmin = mma_half_scale(block + 2);
    const unsigned char *scales = block + 4;
    const unsigned char *qh = block + 16;
    const unsigned char *qs = block + 48;
    unsigned int chunk = within / 64u;
    unsigned int within_chunk = within % 64u;
    bool low = within_chunk < 32u;
    unsigned int sub_block = 2u * chunk + (low ? 0u : 1u);
    unsigned int offset = within_chunk % 32u;
    *scale = d * (float)q4k_scale(scales, sub_block);
    *minimum = dmin * (float)q4k_min(scales, sub_block);
    unsigned int mask = low ? (1u << (2u * chunk)) : (2u << (2u * chunk));
    int levels[8];
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        unsigned int qs_byte = qs[chunk * 32u + offset + slot];
        int nibble = (int)(low ? (qs_byte & 0x0Fu) : (qs_byte >> 4u));
        levels[slot] = nibble + (((unsigned int)qh[offset + slot] & mask) != 0u ? 16 : 0);
    }
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) { out[pair] = mma_pair(levels[2 * pair], levels[2 * pair + 1]); }
}
__device__ __forceinline__ void mma_decode_q6k(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    float d = mma_half_scale(block + 208);
    unsigned int half_index = within / 128u;
    unsigned int local = within % 128u;
    unsigned int l = local % 32u;
    unsigned int lane = local / 32u;
    const unsigned char *ql = block + half_index * 64u;
    const unsigned char *qh = block + 128u + half_index * 32u;
    *scale = d * (float)(signed char)block[192u + half_index * 8u + (l / 16u) + lane * 2u];
    *minimum = 0.0f;
    int levels[8];
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        unsigned int ql_byte = (lane % 2u == 0u) ? ql[l + slot] : ql[l + slot + 32u];
        unsigned int nibble = (lane < 2u) ? (ql_byte & 0x0Fu) : (ql_byte >> 4u);
        unsigned int high2 = ((unsigned int)qh[l + slot] >> (lane * 2u)) & 0x03u;
        levels[slot] = (int)(nibble | (high2 << 4u)) - 32;
    }
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) { out[pair] = mma_pair(levels[2 * pair], levels[2 * pair + 1]); }
}
__device__ __forceinline__ void mma_decode_q1_0(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    *scale = mma_half_scale(block);
    *minimum = 0.0f;
    unsigned int bits = (unsigned int)block[2u + (within >> 3u)];
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        out[pair] = mma_pair(((bits >> (2 * pair)) & 1u) ? 1 : -1, ((bits >> (2 * pair + 1)) & 1u) ? 1 : -1);
    }
}
__device__ __forceinline__ void mma_decode_q2_0(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    *scale = mma_half_scale(block);
    *minimum = 0.0f;
    unsigned int codes = (unsigned int)block[2u + (within >> 2u)] | ((unsigned int)block[3u + (within >> 2u)] << 8u);
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        out[pair] = mma_pair((int)((codes >> (4 * pair)) & 3u) - 1, (int)((codes >> (4 * pair + 2)) & 3u) - 1);
    }
}

__device__ __forceinline__ void mma_s8(int *acc, int fragment_a, int fragment_b) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(acc[0]), "+r"(acc[1]) : "r"(fragment_a), "r"(fragment_b));
}

__device__ __forceinline__ float blockdot_e32_b34_c8sr0_at16_m1_z0_s5x10at0_scale_code(unsigned int code) {
    return __half2float(__ushort_as_half((unsigned short)code));
}
__device__ __forceinline__ float blockdot_e32_b34_c8sr0_at16_m1_z0_s5x10at0_scale(const unsigned char *block) {
    return blockdot_e32_b34_c8sr0_at16_m1_z0_s5x10at0_scale_code((unsigned int)*(const unsigned short *)(block + 0u));
}
__device__ __forceinline__ void blockdot_e32_b34_c8sr0_at16_m1_z0_s5x10at0_i8x32(const unsigned char *block, unsigned int chunk, int *words) {
    const unsigned short *halves = (const unsigned short *)(block + 2u + chunk * 32u);
    #pragma unroll
    for (int word = 0; word < 8; word++) {
        unsigned int packed = (unsigned int)halves[2 * word] | ((unsigned int)halves[2 * word + 1] << 16);
        words[word] = packed;
    }
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

extern "C" __global__ void __launch_bounds__(256, 2) omega_cuda_reduce_r3_ax0_1_n2_multiply_add_zero_wide_cd8_mma8x8x16s8c4_gid64c9(
    const unsigned char* __restrict__ in0,
    const unsigned char* __restrict__ in1,
    float* __restrict__ out,
    const Uniforms* __restrict__ u_ptr
) {
    const Uniforms u = *u_ptr;
    long gid = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int quad = lane & 3;
    const int warp_feature = warp & 3;
    const int warp_token = warp >> 2;
    long feature_total = 1;
    feature_total *= u.output_extents[1];
    long token_total = 1;
    token_total *= u.output_extents[0];
    const long feature_tiles = (feature_total + 128 - 1) / 128;
    const long feature_first = ((long)blockIdx.x % feature_tiles) * 128;
    const long token_first = ((long)blockIdx.x / feature_tiles) * 64;
    __shared__ __align__(16) signed char weight_tile[128][144];
    __shared__ __align__(16) signed char act_tile[64][144];
    __shared__ float weight_scales[4][128];
    __shared__ float act_scales[64][4];
    float flat[32];
    #pragma unroll
    for (int index = 0; index < 32; index++) { flat[index] = 0.0f; }
    for (long slab_first = 0; slab_first < u.reduction_total; slab_first += 128) {
        for (int item = tid; item < 512; item += 256) {
            const int feature_local = item / 4;
            const int chunk = item % 4;
            const long feature = feature_first + feature_local;
            int words[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            float block_scale = 0.0f;
            if (feature < feature_total) {
                const long element = u.operand_base[1] + feature * u.reduction_total + slab_first + chunk * 32;
                const unsigned char *block = in1 + (element / 32) * 34;
                blockdot_e32_b34_c8sr0_at16_m1_z0_s5x10at0_i8x32(block, (unsigned int)(element % 32) / 32u, words);
                block_scale = blockdot_e32_b34_c8sr0_at16_m1_z0_s5x10at0_scale(block);
            }
            uint4 *destination = (uint4 *)&weight_tile[feature_local][chunk * 32];
            destination[0] = make_uint4((unsigned int)words[0], (unsigned int)words[1], (unsigned int)words[2], (unsigned int)words[3]);
            destination[1] = make_uint4((unsigned int)words[4], (unsigned int)words[5], (unsigned int)words[6], (unsigned int)words[7]);
            weight_scales[chunk][feature_local] = block_scale;
        }
        for (int item = tid; item < 256; item += 256) {
            const int token_local = item / 4;
            const int chunk = item % 4;
            const long token = token_first + token_local;
            uint4 low = make_uint4(0u, 0u, 0u, 0u);
            uint4 high = make_uint4(0u, 0u, 0u, 0u);
            float step = 0.0f;
            if (token < token_total) {
                const long group = token * (u.reduction_total / 32) + slab_first / 32 + chunk;
                const unsigned char *bytes = in0 + ((group) / 4) * 144;
                const uint4 activation_low = ((const uint4 *)(bytes + 16 + ((group) % 4) * 32))[0];
                const uint4 activation_high = ((const uint4 *)(bytes + 16 + ((group) % 4) * 32))[1];
                low = activation_low;
                high = activation_high;
                step = mma_half_scale(bytes + 0 + ((group) % 4) * 4);
            }
            uint4 *destination = (uint4 *)&act_tile[token_local][chunk * 32];
            destination[0] = low;
            destination[1] = high;
            act_scales[token_local][chunk] = step;
        }
        __syncthreads();
        #pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            int accumulators[4][4][2] = {};
            #pragma unroll
            for (int step = 0; step < 2; step++) {
                const int depth_at = chunk * 32 + step * 16 + quad * 4;
                int fragment_a[4];
                int fragment_b[4];
                #pragma unroll
                for (int tile = 0; tile < 4; tile++) {
                    fragment_a[tile] = *(const int *)&weight_tile[warp_feature * 32 + tile * 8 + group][depth_at];
                    fragment_b[tile] = *(const int *)&act_tile[warp_token * 32 + tile * 8 + group][depth_at];
                }
                #pragma unroll
                for (int row = 0; row < 4; row++) {
                    #pragma unroll
                    for (int col = 0; col < 4; col++) { mma_s8(accumulators[row][col], fragment_a[row], fragment_b[col]); }
                }
            }
            #pragma unroll
            for (int row = 0; row < 4; row++) {
                const float weight_step = weight_scales[chunk][warp_feature * 32 + row * 8 + group];
                #pragma unroll
                for (int col = 0; col < 4; col++) {
                    #pragma unroll
                    for (int pair = 0; pair < 2; pair++) {
                        const float scale = weight_step * act_scales[warp_token * 32 + col * 8 + quad * 2 + pair][chunk];
                        flat[(row * 4 + col) * 2 + pair] += scale * (float)accumulators[row][col][pair];
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll 1
    for (int index = 0; index < 32; index++) {
        const int row = index >> 3;
        const int col = (index >> 1) & 3;
        const int pair = index & 1;
        const long feature = feature_first + warp_feature * 32 + row * 8 + group;
        const long token = token_first + warp_token * 32 + col * 8 + quad * 2 + pair;
        if (feature >= feature_total || token >= token_total) { continue; }
        long output_coord[2];
        long feature_rest = feature;
        long token_rest = token;
        output_coord[1] = feature_rest % u.output_extents[1]; feature_rest /= u.output_extents[1];
        output_coord[0] = token_rest % u.output_extents[0]; token_rest /= u.output_extents[0];
        long out_offset = u.out_base;
        out_offset += output_coord[0] * u.out_strides[0];
        out_offset += output_coord[1] * u.out_strides[1];
        out[out_offset] = flat[index];
    }
}
