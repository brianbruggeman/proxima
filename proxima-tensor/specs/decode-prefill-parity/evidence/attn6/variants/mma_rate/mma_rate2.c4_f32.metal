#include <metal_stdlib>
using namespace metal;
struct Uniforms { long total_elements; long splits; };
kernel void omega_cached_attention_h8_g2_d64_s3c800000_ln9223372036854775808_up0_r8_n2_b64_rt(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const float* in2 [[buffer(2)]], device const float* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const float* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], uint tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort simdgroup_slot [[simdgroup_index_in_threadgroup]]) {
    simdgroup_float8x8 left_wide; simdgroup_float8x8 right_wide;
    simdgroup_load(left_wide, in0, 8); simdgroup_load(right_wide, in1, 8);
    simdgroup_float8x8 left; simdgroup_float8x8 right;
    left.thread_elements()[0] = float(left_wide.thread_elements()[0]); left.thread_elements()[1] = float(left_wide.thread_elements()[1]);
    right.thread_elements()[0] = float(right_wide.thread_elements()[0]); right.thread_elements()[1] = float(right_wide.thread_elements()[1]);
    simdgroup_float8x8 a0 = make_filled_simdgroup_matrix<float, 8, 8>(0); simdgroup_float8x8 a1 = a0; simdgroup_float8x8 a2 = a0; simdgroup_float8x8 a3 = a0;
    #pragma clang loop unroll(disable)
    for (int step = 0; step < 2048; step++) {
         simdgroup_multiply_accumulate(a0, left, right, a0); simdgroup_multiply_accumulate(a1, left, right, a1); simdgroup_multiply_accumulate(a2, left, right, a2); simdgroup_multiply_accumulate(a3, left, right, a3);
    }
    simdgroup_float8x8 stored;
    stored.thread_elements()[0] = 0.0f  + float(a0.thread_elements()[0]) + float(a1.thread_elements()[0]) + float(a2.thread_elements()[0]) + float(a3.thread_elements()[0]);
    stored.thread_elements()[1] = 0.0f  + float(a0.thread_elements()[1]) + float(a1.thread_elements()[1]) + float(a2.thread_elements()[1]) + float(a3.thread_elements()[1]);
    simdgroup_store(stored, out + ((long)tgid * 2 + simdgroup_slot) * 64, 8);
}
