//! Rootcause round 3, standalone replay + diagnostic-store harness for the
//! gemma4-E2B decode-step-1/layer-0/node-166/dim-1 divergence: fused Metal
//! gives `0x3e0b0e8c`, the unfused 28-op chain gives `0x3e0b0e8b`. This file
//! compiles the PRODUCTION fused kernel MSL verbatim (dumped by
//! `decode.rs::write_attn_layer0_failure_report` into the round-3 fixture
//! manifest) against the same real operand bytes the production plan bound,
//! then a second copy of that same kernel with score/sum/weighted-value
//! store-only instrumentation added, so the per-key `score(t)` the fused
//! kernel actually computes on real Metal hardware can be diffed against the
//! real per-`t` values `payload_r2_safe2.txt` already captured off the
//! UNFUSED plan's own reduce-fold nodes (142 cached / 149 new).
//!
//! # Run
//! ```sh
//! CARGO_TARGET_DIR=<scratch>/target-attn-fuse ATTN_FIXTURE_MANIFEST=<r3>/fixture/manifest.txt \
//!     cargo run -p omega --release --features metal --example attn_fused_replay
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    imp::run();
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    println!("attn_fused_replay requires --features metal on macOS");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod imp {

use core::ffi::c_void;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

// verbatim from the round-3 fixture's `fused_production.metal` dump
// (`omega::emit` on the production `BoundOpKind::CachedAttention`), minus
// the unrelated quantization-decode preamble this entry never calls.
const PRODUCTION_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Uniforms { long total_elements; };

kernel void omega_cached_attention_q1_c32_n1_h1_g8_d256_s3f800000_ln511_up0_cb(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const float* in2 [[buffer(2)]], device const float* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const float* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], uint gid [[thread_position_in_grid]]) {
    if ((long)gid >= u.total_elements * 32L) { return; }
    long cached_key_rows = (long)in8[0]; constexpr long new_key_rows = 1; constexpr long kv_heads = 1; constexpr long query_groups = 8; constexpr long head_dim = 256; constexpr float scale = 1.0; constexpr long cached_lower = -511L; constexpr long new_upper = 0L;
    long vector_index = (long)gid / 32L; uint lane = gid % 32u;
    if (vector_index >= u.total_elements) { return; }
    constexpr long context_chunks = 3;
    long query_index = vector_index / context_chunks;
    long chunk = vector_index % context_chunks;
    long query_row = query_index / (kv_heads * query_groups);
    long remainder = query_index % (kv_heads * query_groups);
    long kv_head = remainder / query_groups;
    long group = remainder % query_groups;
    long query_head = kv_head * query_groups + group;
    long qbase = query_row * (kv_heads * query_groups * (head_dim / 2)) + query_head * (head_dim / 2);
    long local_group_index = group * context_chunks + chunk;
    float maximum = -INFINITY; float sum = 0.0f; float weighted[(head_dim + 31) / 32];
    for (long dimension = 0; dimension < (head_dim + 31) / 32; dimension++) { weighted[dimension] = 0.0f; }
    long last_key = cached_key_rows + new_key_rows - 1L;
    for (long key = chunk; key <= last_key; key += context_chunks) {
        bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;
        long relative = (cached ? key - cached_key_rows : new_index) - query_row;
        if (cached && relative < cached_lower) { continue; }
        if (!cached && relative > new_upper) { continue; }
        long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);
        float partial_score = 0.0f;
        for (long pair = (long)lane; pair < head_dim / 2; pair += 32L) {
            partial_score += in0[qbase + pair] * (cached ? in2[kbase + pair] : in4[kbase + pair]);
            partial_score += in1[qbase + pair] * (cached ? in3[kbase + pair] : in5[kbase + pair]);
        }
        float score = simd_broadcast_first(simd_sum(partial_score)) * scale;
        float next_max = max(maximum, score);
        float weight = exp(score - next_max); float rescale = (maximum == -INFINITY) ? 0.0f : exp(maximum - next_max);
        sum = sum * rescale + weight;
        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
            long local_dimension = dimension / 32L;
            weighted[local_dimension] = weighted[local_dimension] * rescale + weight * (cached ? in6[kbase * 2 + dimension] : in7[kbase * 2 + dimension]);
        }
        maximum = next_max;
    }
    threadgroup float shared_m[query_groups * context_chunks]; threadgroup float shared_l[query_groups * context_chunks]; threadgroup float shared_o[query_groups * context_chunks * head_dim];
    if (lane == 0u) { shared_m[local_group_index] = maximum; shared_l[local_group_index] = sum; }
    for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) { shared_o[local_group_index * head_dim + dimension] = weighted[dimension / 32L]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (chunk == 0L) {
        float merged_max = -INFINITY;
        for (long c = 0; c < context_chunks; c++) { merged_max = max(merged_max, shared_m[group * context_chunks + c]); }
        float merged_sum = 0.0f;
        for (long c = 0; c < context_chunks; c++) {
            float partial_max = shared_m[group * context_chunks + c];
            float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);
            merged_sum += shared_l[group * context_chunks + c] * rescale;
        }
        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
            long local_dimension = dimension / 32L;
            float acc = 0.0f;
            for (long c = 0; c < context_chunks; c++) {
                float partial_max = shared_m[group * context_chunks + c];
                float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);
                acc += shared_o[(group * context_chunks + c) * head_dim + dimension] * rescale;
            }
            weighted[local_dimension] = acc;
        }
        sum = merged_sum;
        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) { long local_dimension = dimension / 32L; out[query_index * head_dim + dimension] = (float)(sum == 0.0f ? 0.0f : weighted[local_dimension] / sum); }
    }
}
"#;

// same body, store-only: `diag[key]` = this simdgroup's own `score(t)` right
// after `simd_sum`/`* scale`, one write per live key, keyed by `key` itself
// (unambiguous across chunks since every key is visited by exactly one
// chunk) -- no arithmetic in the kernel changes, only the two `diag[...] =`
// lines and the added buffer parameter.
const DIAGNOSTIC_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Uniforms { long total_elements; };

kernel void diag_cached_attention(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const float* in2 [[buffer(2)]], device const float* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const float* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], device float* diag [[buffer(11)]], uint gid [[thread_position_in_grid]]) {
    if ((long)gid >= u.total_elements * 32L) { return; }
    long cached_key_rows = (long)in8[0]; constexpr long new_key_rows = 1; constexpr long kv_heads = 1; constexpr long query_groups = 8; constexpr long head_dim = 256; constexpr float scale = 1.0; constexpr long cached_lower = -511L; constexpr long new_upper = 0L;
    long vector_index = (long)gid / 32L; uint lane = gid % 32u;
    if (vector_index >= u.total_elements) { return; }
    constexpr long context_chunks = 3;
    long query_index = vector_index / context_chunks;
    long chunk = vector_index % context_chunks;
    long query_row = query_index / (kv_heads * query_groups);
    long remainder = query_index % (kv_heads * query_groups);
    long kv_head = remainder / query_groups;
    long group = remainder % query_groups;
    long query_head = kv_head * query_groups + group;
    long qbase = query_row * (kv_heads * query_groups * (head_dim / 2)) + query_head * (head_dim / 2);
    long local_group_index = group * context_chunks + chunk;
    float maximum = -INFINITY; float sum = 0.0f; float weighted[(head_dim + 31) / 32];
    for (long dimension = 0; dimension < (head_dim + 31) / 32; dimension++) { weighted[dimension] = 0.0f; }
    long last_key = cached_key_rows + new_key_rows - 1L;
    for (long key = chunk; key <= last_key; key += context_chunks) {
        bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;
        long relative = (cached ? key - cached_key_rows : new_index) - query_row;
        if (cached && relative < cached_lower) { continue; }
        if (!cached && relative > new_upper) { continue; }
        long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);
        float partial_score = 0.0f;
        for (long pair = (long)lane; pair < head_dim / 2; pair += 32L) {
            partial_score += in0[qbase + pair] * (cached ? in2[kbase + pair] : in4[kbase + pair]);
            partial_score += in1[qbase + pair] * (cached ? in3[kbase + pair] : in5[kbase + pair]);
        }
        float score = simd_broadcast_first(simd_sum(partial_score)) * scale;
        if (group == 0 && lane == 0u) { diag[key] = score; }
        float next_max = max(maximum, score);
        float weight = exp(score - next_max); float rescale = (maximum == -INFINITY) ? 0.0f : exp(maximum - next_max);
        sum = sum * rescale + weight;
        if (group == 0 && lane == 0u) { diag[16 + key] = next_max; diag[32 + key] = weight; diag[48 + key] = rescale; diag[64 + key] = sum; }
        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
            long local_dimension = dimension / 32L;
            weighted[local_dimension] = weighted[local_dimension] * rescale + weight * (cached ? in6[kbase * 2 + dimension] : in7[kbase * 2 + dimension]);
            if (group == 0 && dimension == 0) { diag[80 + key] = weighted[local_dimension]; }
            if (group == 0 && dimension == 1) { diag[96 + key] = weighted[local_dimension]; }
        }
        maximum = next_max;
    }
    threadgroup float shared_m[query_groups * context_chunks]; threadgroup float shared_l[query_groups * context_chunks]; threadgroup float shared_o[query_groups * context_chunks * head_dim];
    if (lane == 0u) { shared_m[local_group_index] = maximum; shared_l[local_group_index] = sum; }
    for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) { shared_o[local_group_index * head_dim + dimension] = weighted[dimension / 32L]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (chunk == 0L) {
        float merged_max = -INFINITY;
        for (long c = 0; c < context_chunks; c++) { merged_max = max(merged_max, shared_m[group * context_chunks + c]); }
        float merged_sum = 0.0f;
        for (long c = 0; c < context_chunks; c++) {
            float partial_max = shared_m[group * context_chunks + c];
            float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);
            merged_sum += shared_l[group * context_chunks + c] * rescale;
        }
        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
            long local_dimension = dimension / 32L;
            float acc = 0.0f;
            for (long c = 0; c < context_chunks; c++) {
                float partial_max = shared_m[group * context_chunks + c];
                float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);
                acc += shared_o[(group * context_chunks + c) * head_dim + dimension] * rescale;
            }
            weighted[local_dimension] = acc;
        }
        sum = merged_sum;
        if (group == 0 && lane == 0u) { diag[112] = merged_sum; }
        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
            long local_dimension = dimension / 32L;
            if (group == 0 && dimension == 0) { diag[113] = weighted[local_dimension]; }
            if (group == 0 && dimension == 1) { diag[114] = weighted[local_dimension]; }
            out[query_index * head_dim + dimension] = (float)(sum == 0.0f ? 0.0f : weighted[local_dimension] / sum);
        }
    }
}
"#;

// identical to `DIAGNOSTIC_KERNEL` except the dot-product accumulation: two
// independent running sums (`partial_even`/`partial_odd`), each `simd_sum`ed
// on its own, added together AFTER the reduction -- the unfused chain's own
// association (node=139 odd-dot reduce, node=142 even-dot reduce, epilogue
// `add`), instead of one interleaved per-lane running sum reduced once.
const TOGGLE_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Uniforms { long total_elements; };

kernel void toggle_cached_attention(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const float* in2 [[buffer(2)]], device const float* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const float* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], device float* diag [[buffer(11)]], uint gid [[thread_position_in_grid]]) {
    if ((long)gid >= u.total_elements * 32L) { return; }
    long cached_key_rows = (long)in8[0]; constexpr long new_key_rows = 1; constexpr long kv_heads = 1; constexpr long query_groups = 8; constexpr long head_dim = 256; constexpr float scale = 1.0; constexpr long cached_lower = -511L; constexpr long new_upper = 0L;
    long vector_index = (long)gid / 32L; uint lane = gid % 32u;
    if (vector_index >= u.total_elements) { return; }
    constexpr long context_chunks = 3;
    long query_index = vector_index / context_chunks;
    long chunk = vector_index % context_chunks;
    long query_row = query_index / (kv_heads * query_groups);
    long remainder = query_index % (kv_heads * query_groups);
    long kv_head = remainder / query_groups;
    long group = remainder % query_groups;
    long query_head = kv_head * query_groups + group;
    long qbase = query_row * (kv_heads * query_groups * (head_dim / 2)) + query_head * (head_dim / 2);
    long local_group_index = group * context_chunks + chunk;
    float maximum = -INFINITY; float sum = 0.0f; float weighted[(head_dim + 31) / 32];
    for (long dimension = 0; dimension < (head_dim + 31) / 32; dimension++) { weighted[dimension] = 0.0f; }
    long last_key = cached_key_rows + new_key_rows - 1L;
    for (long key = chunk; key <= last_key; key += context_chunks) {
        bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;
        long relative = (cached ? key - cached_key_rows : new_index) - query_row;
        if (cached && relative < cached_lower) { continue; }
        if (!cached && relative > new_upper) { continue; }
        long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);
        float partial_even = 0.0f; float partial_odd = 0.0f;
        for (long pair = (long)lane; pair < head_dim / 2; pair += 32L) {
            partial_even += in0[qbase + pair] * (cached ? in2[kbase + pair] : in4[kbase + pair]);
            partial_odd += in1[qbase + pair] * (cached ? in3[kbase + pair] : in5[kbase + pair]);
        }
        float score = (simd_broadcast_first(simd_sum(partial_even)) + simd_broadcast_first(simd_sum(partial_odd))) * scale;
        if (group == 0 && lane == 0u) { diag[key] = score; }
        float next_max = max(maximum, score);
        float weight = exp(score - next_max); float rescale = (maximum == -INFINITY) ? 0.0f : exp(maximum - next_max);
        sum = sum * rescale + weight;
        if (group == 0 && lane == 0u) { diag[16 + key] = next_max; diag[32 + key] = weight; diag[48 + key] = rescale; diag[64 + key] = sum; }
        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
            long local_dimension = dimension / 32L;
            weighted[local_dimension] = weighted[local_dimension] * rescale + weight * (cached ? in6[kbase * 2 + dimension] : in7[kbase * 2 + dimension]);
            if (group == 0 && dimension == 0) { diag[80 + key] = weighted[local_dimension]; }
            if (group == 0 && dimension == 1) { diag[96 + key] = weighted[local_dimension]; }
        }
        maximum = next_max;
    }
    threadgroup float shared_m[query_groups * context_chunks]; threadgroup float shared_l[query_groups * context_chunks]; threadgroup float shared_o[query_groups * context_chunks * head_dim];
    if (lane == 0u) { shared_m[local_group_index] = maximum; shared_l[local_group_index] = sum; }
    for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) { shared_o[local_group_index * head_dim + dimension] = weighted[dimension / 32L]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (chunk == 0L) {
        float merged_max = -INFINITY;
        for (long c = 0; c < context_chunks; c++) { merged_max = max(merged_max, shared_m[group * context_chunks + c]); }
        float merged_sum = 0.0f;
        for (long c = 0; c < context_chunks; c++) {
            float partial_max = shared_m[group * context_chunks + c];
            float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);
            merged_sum += shared_l[group * context_chunks + c] * rescale;
        }
        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
            long local_dimension = dimension / 32L;
            float acc = 0.0f;
            for (long c = 0; c < context_chunks; c++) {
                float partial_max = shared_m[group * context_chunks + c];
                float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);
                acc += shared_o[(group * context_chunks + c) * head_dim + dimension] * rescale;
            }
            weighted[local_dimension] = acc;
        }
        sum = merged_sum;
        if (group == 0 && lane == 0u) { diag[112] = merged_sum; }
        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
            long local_dimension = dimension / 32L;
            if (group == 0 && dimension == 0) { diag[113] = weighted[local_dimension]; }
            if (group == 0 && dimension == 1) { diag[114] = weighted[local_dimension]; }
            out[query_index * head_dim + dimension] = (float)(sum == 0.0f ? 0.0f : weighted[local_dimension] / sum);
        }
    }
}
"#;

// round-4 work item B: dumps EVERY lane's own `partial_even`/`partial_odd`
// for key=2 (the residual key), group=0, plus TWO independently-computed
// totals of those SAME 32 real lane values: `simd_sum` (the hardware
// reduction both TOGGLE_KERNEL and the real unfused node=139/142 kernels
// use) and a manual sequential lane-0 walk over threadgroup memory (the
// "natural" left-to-right order) -- isolates whether `simd_sum`'s own
// internal tree is the source of the residual, independent of association.
const LANE_DIAG_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Uniforms { long total_elements; };

kernel void lane_diag_cached_attention(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const float* in2 [[buffer(2)]], device const float* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const float* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], device float* diag [[buffer(11)]], uint gid [[thread_position_in_grid]]) {
    if ((long)gid >= u.total_elements * 32L) { return; }
    long cached_key_rows = (long)in8[0]; constexpr long kv_heads = 1; constexpr long query_groups = 8; constexpr long head_dim = 256;
    long vector_index = (long)gid / 32L; uint lane = gid % 32u;
    if (vector_index >= u.total_elements) { return; }
    constexpr long context_chunks = 3;
    long query_index = vector_index / context_chunks;
    long chunk = vector_index % context_chunks;
    long query_row = query_index / (kv_heads * query_groups);
    long remainder = query_index % (kv_heads * query_groups);
    long kv_head = remainder / query_groups;
    long group = remainder % query_groups;
    long query_head = kv_head * query_groups + group;
    long qbase = query_row * (kv_heads * query_groups * (head_dim / 2)) + query_head * (head_dim / 2);
    threadgroup float lane_even[32]; threadgroup float lane_odd[32];
    constexpr long key = 2;
    if (chunk == key % context_chunks) {
        bool cached = key < cached_key_rows;
        long kbase = key * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);
        float partial_even = 0.0f; float partial_odd = 0.0f;
        for (long pair = (long)lane; pair < head_dim / 2; pair += 32L) {
            partial_even += in0[qbase + pair] * (cached ? in2[kbase + pair] : in4[kbase + pair]);
            partial_odd += in1[qbase + pair] * (cached ? in3[kbase + pair] : in5[kbase + pair]);
        }
        if (group == 0) {
            diag[lane] = partial_even;
            diag[32 + lane] = partial_odd;
            lane_even[lane] = partial_even;
            lane_odd[lane] = partial_odd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (group == 0 && lane == 0u) {
            float manual_even = 0.0f; float manual_odd = 0.0f;
            for (uint index = 0; index < 32u; index++) { manual_even += lane_even[index]; manual_odd += lane_odd[index]; }
            diag[64] = manual_even;
            diag[65] = manual_odd;
        }
        float simd_even = simd_sum(partial_even);
        float simd_odd = simd_sum(partial_odd);
        if (group == 0 && lane == 0u) {
            diag[66] = simd_even;
            diag[67] = simd_odd;
        }
    }
}
"#;

fn parse_leaf(manifest: &str, label: &str) -> Vec<f32> {
    let marker = format!("leaf_full label={label} ");
    let line = manifest
        .lines()
        .find(|line| line.starts_with(&marker))
        .unwrap_or_else(|| panic!("fixture manifest missing leaf_full label={label}"));
    let bits_start = line.find("bits=[").expect("leaf_full line carries bits=[...]") + "bits=[".len();
    let bits_end = line.rfind(']').expect("leaf_full line's bits array is closed");
    line[bits_start..bits_end]
        .split(", ")
        .map(|token| f32::from_bits(token.parse::<u32>().expect("leaf bit token parses as u32")))
        .collect()
}

fn compile(device: &ProtocolObject<dyn MTLDevice>, source: &str, entry: &str, mode: MTLMathMode) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let options = MTLCompileOptions::new();
    options.setMathMode(mode);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .unwrap_or_else(|error| panic!("compiles {entry}: {}", error.localizedDescription()));
    let function = library
        .newFunctionWithName(&NSString::from_str(entry))
        .unwrap_or_else(|| panic!("kernel entry `{entry}` missing from its own compiled library"));
    device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|error| panic!("creates the pipeline for {entry}: {}", error.localizedDescription()))
}

fn shared_buffer(device: &ProtocolObject<dyn MTLDevice>, bytes: &[u8]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let pointer = unsafe { NonNull::new_unchecked(bytes.as_ptr() as *mut c_void) };
    unsafe { device.newBufferWithBytes_length_options(pointer, bytes.len(), MTLResourceOptions::StorageModeShared) }
        .expect("device allocates a fresh shared buffer copied from real fixture bytes")
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn read_f32_buffer(buffer: &ProtocolObject<dyn MTLBuffer>, count: usize) -> Vec<f32> {
    let pointer = buffer.contents().as_ptr().cast::<f32>();
    unsafe { core::slice::from_raw_parts(pointer, count) }.to_vec()
}

fn dispatch(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[&ProtocolObject<dyn MTLBuffer>],
) {
    let command_buffer = queue.commandBuffer().expect("command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("compute encoder");
    encoder.setComputePipelineState(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        unsafe { encoder.setBuffer_offset_atIndex(Some(*buffer), 0, index) };
    }
    let grid = MTLSize { width: 768, height: 1, depth: 1 };
    let threadgroup = MTLSize { width: 768, height: 1, depth: 1 };
    encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let _ = device;
}

pub fn run() {
    let manifest_path = std::env::var("ATTN_FIXTURE_MANIFEST")
        .expect("ATTN_FIXTURE_MANIFEST must point at the round-3 fixture manifest.txt");
    let manifest = std::fs::read_to_string(&manifest_path).expect("read fixture manifest");

    let q_even = parse_leaf(&manifest, "q_even");
    let q_odd = parse_leaf(&manifest, "q_odd");
    let k_even_cache = parse_leaf(&manifest, "k_even_cache");
    let k_odd_cache = parse_leaf(&manifest, "k_odd_cache");
    let new_k_even = parse_leaf(&manifest, "new_k_even");
    let new_k_odd = parse_leaf(&manifest, "new_k_odd");
    let v_cache = parse_leaf(&manifest, "v_cache");
    let v_new = parse_leaf(&manifest, "v_new");
    let cached_len = [6.0_f32];
    let uniforms_total_elements: i64 = 24;

    println!(
        "attn_fused_replay fixture_loaded q_even={} q_odd={} k_even_cache={} k_odd_cache={} new_k_even={} new_k_odd={} v_cache={} v_new={}",
        q_even.len(), q_odd.len(), k_even_cache.len(), k_odd_cache.len(), new_k_even.len(), new_k_odd.len(), v_cache.len(), v_new.len(),
    );

    let device = MTLCreateSystemDefaultDevice().expect("system default Metal device");
    let queue = device.newCommandQueue().expect("command queue");

    let in0 = shared_buffer(&device, &f32_bytes(&q_even));
    let in1 = shared_buffer(&device, &f32_bytes(&q_odd));
    let in2 = shared_buffer(&device, &f32_bytes(&k_even_cache));
    let in3 = shared_buffer(&device, &f32_bytes(&k_odd_cache));
    let in4 = shared_buffer(&device, &f32_bytes(&new_k_even));
    let in5 = shared_buffer(&device, &f32_bytes(&new_k_odd));
    let in6 = shared_buffer(&device, &f32_bytes(&v_cache));
    let in7 = shared_buffer(&device, &f32_bytes(&v_new));
    let in8 = shared_buffer(&device, &f32_bytes(&cached_len));
    let uniforms = shared_buffer(&device, &uniforms_total_elements.to_le_bytes());

    // round-4 work item A: `ATTN_FULL_KERNEL_SOURCE`, when set, compiles the
    // WHOLE dumped `fused_production.metal` (every unrelated quantization
    // helper included, byte-identical to `kernel.source` production itself
    // compiles) instead of the hand-trimmed `PRODUCTION_KERNEL` literal --
    // isolates whether Relaxed-mode codegen for this one entry is sensitive
    // to the surrounding translation unit, since `MTLCompileOptions` itself
    // is already confirmed identical (`compile_pipeline` only ever sets
    // `mathMode`, nothing else -- `omega/src/metal/pipeline_buffers_upload.rs:172-181`).
    let full_module_source = std::env::var("ATTN_FULL_KERNEL_SOURCE")
        .ok()
        .map(|path| std::fs::read_to_string(path).expect("read full production kernel module"));
    // `ATTN_SINGLE_MODE=safe|relaxed` -- compiles and dispatches ONLY that
    // one mode, nothing else, in an otherwise-empty process: the isolation
    // control for the compile-order cross-contamination this file's
    // multi-mode loop turned up (same entry name compiled twice in one
    // `MTLDevice` session; whichever mode compiles FIRST reproduces the
    // real per-mode production bits, the SECOND compile in the same
    // process inherits the first's codegen regardless of its own
    // `mathMode`). A single-process, single-compile run has no "second
    // compile" to contaminate it.
    let mode_order: Vec<(&str, MTLMathMode)> = match std::env::var("ATTN_SINGLE_MODE").ok().as_deref() {
        Some("safe") => vec![("Safe", MTLMathMode::Safe)],
        Some("relaxed") => vec![("Relaxed", MTLMathMode::Relaxed)],
        _ if std::env::var_os("ATTN_RELAXED_FIRST").is_some() => {
            vec![("Relaxed", MTLMathMode::Relaxed), ("Safe", MTLMathMode::Safe)]
        }
        _ => vec![("Safe", MTLMathMode::Safe), ("Relaxed", MTLMathMode::Relaxed)],
    };
    for (mode_name, mode) in mode_order {
        let source = full_module_source.as_deref().unwrap_or(PRODUCTION_KERNEL);
        let pipeline = compile(
            &device,
            source,
            "omega_cached_attention_q1_c32_n1_h1_g8_d256_s3f800000_ln511_up0_cb",
            mode,
        );
        let out = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 2048]));
        dispatch(&device, &queue, &pipeline, &[&in0, &in1, &in2, &in3, &in4, &in5, &in6, &in7, &in8, &out, &uniforms]);
        let out_values = read_f32_buffer(&out, 2048);
        let hash: u64 = out_values[..256].iter().fold(0xcbf29ce484222325_u64, |accumulator, value| {
            (accumulator ^ u64::from(value.to_bits())).wrapping_mul(0x100000001b3)
        });
        println!(
            "attn_fused_replay math_mode={mode_name} out0_bits=0x{:08x} out1_bits=0x{:08x} group0_hash=0x{hash:016x}",
            out_values[0].to_bits(),
            out_values[1].to_bits(),
        );
    }

    // reference per-key masked scores read straight off `payload_r2_safe2.txt`
    // (node=142 qg0_column t=0..5, node=149 qg0_column t=6) -- the UNFUSED
    // plan's real Metal-computed values for the identical operand bytes.
    const REFERENCE_SCORE_BITS: [u32; 7] = [
        1092639986, 1090548168, 1077815366, 1087431027, 1088302311, 1089938621, 1086792084,
    ];

    let diag_pipeline = compile(&device, DIAGNOSTIC_KERNEL, "diag_cached_attention", MTLMathMode::Safe);
    let out = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 2048]));
    let diag = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 128]));
    dispatch(&device, &queue, &diag_pipeline, &[&in0, &in1, &in2, &in3, &in4, &in5, &in6, &in7, &in8, &out, &uniforms, &diag]);
    let out_values = read_f32_buffer(&out, 2048);
    let diag_values = read_f32_buffer(&diag, 128);
    println!(
        "attn_fused_replay diag_out0_bits=0x{:08x} diag_out1_bits=0x{:08x}",
        out_values[0].to_bits(),
        out_values[1].to_bits(),
    );
    for key in 0..7usize {
        let fused_bits = diag_values[key].to_bits();
        let reference_bits = REFERENCE_SCORE_BITS[key];
        println!(
            "attn_fused_replay score t={key} fused=0x{fused_bits:08x} unfused_reference=0x{reference_bits:08x} match={}",
            fused_bits == reference_bits
        );
    }
    println!(
        "attn_fused_replay merged_sum=0x{:08x} weighted_dim0=0x{:08x} weighted_dim1=0x{:08x}",
        diag_values[112].to_bits(),
        diag_values[113].to_bits(),
        diag_values[114].to_bits(),
    );
    for key in 0..7usize {
        println!(
            "attn_fused_replay key={key} next_max=0x{:08x} weight=0x{:08x} rescale=0x{:08x} running_sum=0x{:08x} weighted_dim0=0x{:08x} weighted_dim1=0x{:08x}",
            diag_values[16 + key].to_bits(),
            diag_values[32 + key].to_bits(),
            diag_values[48 + key].to_bits(),
            diag_values[64 + key].to_bits(),
            diag_values[80 + key].to_bits(),
            diag_values[96 + key].to_bits(),
        );
    }

    let toggle_pipeline = compile(&device, TOGGLE_KERNEL, "toggle_cached_attention", MTLMathMode::Safe);
    let toggle_out = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 2048]));
    let toggle_diag = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 128]));
    dispatch(
        &device,
        &queue,
        &toggle_pipeline,
        &[&in0, &in1, &in2, &in3, &in4, &in5, &in6, &in7, &in8, &toggle_out, &uniforms, &toggle_diag],
    );
    let toggle_out_values = read_f32_buffer(&toggle_out, 2048);
    let toggle_diag_values = read_f32_buffer(&toggle_diag, 128);
    println!(
        "attn_fused_replay toggle_out0_bits=0x{:08x} toggle_out1_bits=0x{:08x}",
        toggle_out_values[0].to_bits(),
        toggle_out_values[1].to_bits(),
    );
    for key in 0..7usize {
        let before_bits = diag_values[key].to_bits();
        let after_bits = toggle_diag_values[key].to_bits();
        let reference_bits = REFERENCE_SCORE_BITS[key];
        println!(
            "attn_fused_replay toggle score t={key} before=0x{before_bits:08x} after=0x{after_bits:08x} unfused_reference=0x{reference_bits:08x} after_matches_reference={}",
            after_bits == reference_bits
        );
    }

    let lane_pipeline = compile(&device, LANE_DIAG_KERNEL, "lane_diag_cached_attention", MTLMathMode::Safe);
    let lane_out = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 2048]));
    let lane_diag = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 128]));
    dispatch(
        &device,
        &queue,
        &lane_pipeline,
        &[&in0, &in1, &in2, &in3, &in4, &in5, &in6, &in7, &in8, &lane_out, &uniforms, &lane_diag],
    );
    let lane_diag_values = read_f32_buffer(&lane_diag, 128);
    for lane in 0..32usize {
        println!(
            "attn_fused_replay lane_diag key=2 lane={lane} partial_even=0x{:08x} partial_odd=0x{:08x}",
            lane_diag_values[lane].to_bits(),
            lane_diag_values[32 + lane].to_bits(),
        );
    }
    println!(
        "attn_fused_replay lane_diag key=2 manual_even=0x{:08x} manual_odd=0x{:08x} simd_even=0x{:08x} simd_odd=0x{:08x} manual_even_eq_simd_even={} manual_odd_eq_simd_odd={}",
        lane_diag_values[64].to_bits(),
        lane_diag_values[65].to_bits(),
        lane_diag_values[66].to_bits(),
        lane_diag_values[67].to_bits(),
        lane_diag_values[64].to_bits() == lane_diag_values[66].to_bits(),
        lane_diag_values[65].to_bits() == lane_diag_values[67].to_bits(),
    );
}

}
