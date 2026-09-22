//! Rootcause round 6: FMA-contraction isolation for the gemma4-E2B
//! decode-step-1/layer-0/node-166 key=2 divergence. Round 5 showed the
//! per-lane even-dot accumulation itself (not the 32-lane fold) differs
//! between the real node=142 kernel and the production-shaped loop, but
//! per-step `product` stores used to localize it were themselves shown to
//! change codegen (a store forces the product to round/materialize,
//! defeating FMA contraction). This file replaces per-step GPU stores with
//! a HOST-SIDE f32 replay of both candidate forms (`acc + q*k` separate
//! rounding vs `q.mul_add(k, acc)` fused) driven by the real fixture bits,
//! and captures only each lane's FINAL accumulator (one store, after the
//! loop, before `simd_sum`) from the GPU kernels to compare against.
//!
//! # Run
//! ```sh
//! CARGO_TARGET_DIR=<scratch>/target-attn-fuse ATTN_FIXTURE_MANIFEST=<r4>/fixture/manifest_safe_nodekernels.txt \
//!     cargo run -p omega --release --features metal --example attn_contraction_replay
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    imp::run();
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    println!("attn_contraction_replay requires --features metal on macOS");
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

// node=139 real kernel, FINAL-accumulator-only diagnostic: ONE store per
// lane, after the 4-step loop, before `simd_sum` -- no per-step stores, so
// no product/accumulator is forced to materialize mid-loop.
const NODE139_FINAL_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    long output_total;
    long reduction_total;
    long output_extents[4];
    long reduction_extents[1];
    long operand_base[2];
    long operand_strides[2][5];
    long out_base;
    long out_strides[5];
};

kernel void final_node139(
    device const float* in0 [[buffer(0)]],
    device const float* in1 [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant Uniforms& u [[buffer(3)]],
    device float* diag_final [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    long output_index = (long)gid / 32;
    if (output_index >= u.output_total) { return; }
    uint lane = gid % 32u;
    long full_coord[5];
    full_coord[0] = 0; full_coord[1] = 0; full_coord[2] = 0; full_coord[3] = 0; full_coord[4] = 0;
    long output_coord[4];
    long remaining = output_index;
    output_coord[3] = remaining % u.output_extents[3]; remaining /= u.output_extents[3];
    output_coord[2] = remaining % u.output_extents[2]; remaining /= u.output_extents[2];
    output_coord[1] = remaining % u.output_extents[1]; remaining /= u.output_extents[1];
    output_coord[0] = remaining % u.output_extents[0]; remaining /= u.output_extents[0];
    full_coord[0] = output_coord[0]; full_coord[1] = output_coord[1]; full_coord[2] = output_coord[2]; full_coord[3] = output_coord[3];
    float accumulator;
    bool seeded;
    if (lane == 0u) { accumulator = 0.0f; seeded = true; } else { accumulator = 0.0f; seeded = true; }
    long stride0 = u.operand_strides[0][4];
    long off0 = u.operand_base[0];
    off0 += full_coord[0] * u.operand_strides[0][0];
    off0 += full_coord[1] * u.operand_strides[0][1];
    off0 += full_coord[2] * u.operand_strides[0][2];
    off0 += full_coord[3] * u.operand_strides[0][3];
    off0 += (long)lane * stride0;
    int walk0 = (int)off0;
    int advance0 = (int)(stride0 * 32);
    long stride1 = u.operand_strides[1][4];
    long off1 = u.operand_base[1];
    off1 += full_coord[0] * u.operand_strides[1][0];
    off1 += full_coord[1] * u.operand_strides[1][1];
    off1 += full_coord[2] * u.operand_strides[1][2];
    off1 += full_coord[3] * u.operand_strides[1][3];
    off1 += (long)lane * stride1;
    int walk1 = (int)off1;
    int advance1 = (int)(stride1 * 32);
    for (int r = (int)lane; r < (int)u.reduction_total; r += 32) {
        float scratch[2];
        scratch[0] = in0[walk0];
        scratch[1] = in1[walk1];
        float value = scratch[0] * scratch[1];
        accumulator = seeded ? (accumulator + value) : value;
        seeded = true;
        walk0 += advance0;
        walk1 += advance1;
    }
    if (output_index == 16) { diag_final[lane] = accumulator; }
    float reduced = simd_sum(accumulator);
    if (lane == 0u) {
        long out_offset = u.out_base;
        out_offset += full_coord[0] * u.out_strides[0];
        out_offset += full_coord[1] * u.out_strides[1];
        out_offset += full_coord[2] * u.out_strides[2];
        out_offset += full_coord[3] * u.out_strides[3];
        out_offset += full_coord[4] * u.out_strides[4];
        out[out_offset] = reduced;
    }
}
"#;

// node=142 real kernel, same FINAL-accumulator-only treatment plus the
// unchanged epilogue (add node139 + select mask) so the gate can still
// check its real production output bit.
const NODE142_FINAL_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    long output_total;
    long reduction_total;
    long output_extents[4];
    long reduction_extents[1];
    long operand_base[2];
    long operand_strides[2][5];
    long out_base;
    long out_strides[5];
    long epilogue_operand_base[3];
    long epilogue_operand_strides[3][4];
};

kernel void final_node142(
    device const float* in0 [[buffer(0)]],
    device const float* in1 [[buffer(1)]],
    device const float* epi0 [[buffer(2)]],
    device const float* epi1 [[buffer(3)]],
    device const float* epi2 [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant Uniforms& u [[buffer(6)]],
    device float* diag_final [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    long output_index = (long)gid / 32;
    if (output_index >= u.output_total) { return; }
    uint lane = gid % 32u;
    long full_coord[5];
    full_coord[0] = 0; full_coord[1] = 0; full_coord[2] = 0; full_coord[3] = 0; full_coord[4] = 0;
    long output_coord[4];
    long remaining = output_index;
    output_coord[3] = remaining % u.output_extents[3]; remaining /= u.output_extents[3];
    output_coord[2] = remaining % u.output_extents[2]; remaining /= u.output_extents[2];
    output_coord[1] = remaining % u.output_extents[1]; remaining /= u.output_extents[1];
    output_coord[0] = remaining % u.output_extents[0]; remaining /= u.output_extents[0];
    full_coord[0] = output_coord[0]; full_coord[1] = output_coord[1]; full_coord[2] = output_coord[2]; full_coord[3] = output_coord[3];
    float accumulator;
    bool seeded;
    if (lane == 0u) { accumulator = 0.0f; seeded = true; } else { accumulator = 0.0f; seeded = true; }
    long stride0 = u.operand_strides[0][4];
    long off0 = u.operand_base[0];
    off0 += full_coord[0] * u.operand_strides[0][0];
    off0 += full_coord[1] * u.operand_strides[0][1];
    off0 += full_coord[2] * u.operand_strides[0][2];
    off0 += full_coord[3] * u.operand_strides[0][3];
    off0 += (long)lane * stride0;
    int walk0 = (int)off0;
    int advance0 = (int)(stride0 * 32);
    long stride1 = u.operand_strides[1][4];
    long off1 = u.operand_base[1];
    off1 += full_coord[0] * u.operand_strides[1][0];
    off1 += full_coord[1] * u.operand_strides[1][1];
    off1 += full_coord[2] * u.operand_strides[1][2];
    off1 += full_coord[3] * u.operand_strides[1][3];
    off1 += (long)lane * stride1;
    int walk1 = (int)off1;
    int advance1 = (int)(stride1 * 32);
    for (int r = (int)lane; r < (int)u.reduction_total; r += 32) {
        float scratch[2];
        scratch[0] = in0[walk0];
        scratch[1] = in1[walk1];
        float value = scratch[0] * scratch[1];
        accumulator = seeded ? (accumulator + value) : value;
        seeded = true;
        walk0 += advance0;
        walk1 += advance1;
    }
    if (output_index == 16) { diag_final[lane] = accumulator; }
    float reduced = simd_sum(accumulator);
    if (lane == 0u) {
        long out_offset = u.out_base;
        out_offset += full_coord[0] * u.out_strides[0];
        out_offset += full_coord[1] * u.out_strides[1];
        out_offset += full_coord[2] * u.out_strides[2];
        out_offset += full_coord[3] * u.out_strides[3];
        out_offset += full_coord[4] * u.out_strides[4];
        float epi_scratch[4];
        long epi_off0 = u.epilogue_operand_base[0];
        epi_off0 += output_coord[0] * u.epilogue_operand_strides[0][0];
        epi_off0 += output_coord[1] * u.epilogue_operand_strides[0][1];
        epi_off0 += output_coord[2] * u.epilogue_operand_strides[0][2];
        epi_off0 += output_coord[3] * u.epilogue_operand_strides[0][3];
        epi_scratch[0] = epi0[epi_off0];
        long epi_off1 = u.epilogue_operand_base[1];
        epi_off1 += output_coord[0] * u.epilogue_operand_strides[1][0];
        epi_off1 += output_coord[1] * u.epilogue_operand_strides[1][1];
        epi_off1 += output_coord[2] * u.epilogue_operand_strides[1][2];
        epi_off1 += output_coord[3] * u.epilogue_operand_strides[1][3];
        epi_scratch[1] = epi1[epi_off1];
        long epi_off2 = u.epilogue_operand_base[2];
        epi_off2 += output_coord[0] * u.epilogue_operand_strides[2][0];
        epi_off2 += output_coord[1] * u.epilogue_operand_strides[2][1];
        epi_off2 += output_coord[2] * u.epilogue_operand_strides[2][2];
        epi_off2 += output_coord[3] * u.epilogue_operand_strides[2][3];
        epi_scratch[2] = epi2[epi_off2];
        epi_scratch[3] = reduced;
        float epi_step0 = epi_scratch[3];
        float epi_step1 = (epi_step0 + epi_scratch[2]);
        float epi_step2 = ((epi_scratch[0] != 0.0f) ? epi_scratch[1] : epi_step1);
        out[out_offset] = epi_step2;
    }
}
"#;

// same as `NODE142_FINAL_KERNEL` with `#pragma METAL fp contract(off)`
// prepended -- the converse control: if production node=142 (Safe mode,
// contraction ON by Metal's default even under Safe) is itself relying on
// FMA contraction for its captured bits, forcing contraction off here will
// change its output away from the production reference.
const NODE142_FINAL_KERNEL_CONTRACT_OFF: &str = r#"
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    long output_total;
    long reduction_total;
    long output_extents[4];
    long reduction_extents[1];
    long operand_base[2];
    long operand_strides[2][5];
    long out_base;
    long out_strides[5];
    long epilogue_operand_base[3];
    long epilogue_operand_strides[3][4];
};

kernel void final_node142_contract_off(
    device const float* in0 [[buffer(0)]],
    device const float* in1 [[buffer(1)]],
    device const float* epi0 [[buffer(2)]],
    device const float* epi1 [[buffer(3)]],
    device const float* epi2 [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant Uniforms& u [[buffer(6)]],
    device float* diag_final [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    long output_index = (long)gid / 32;
    if (output_index >= u.output_total) { return; }
    uint lane = gid % 32u;
    long full_coord[5];
    full_coord[0] = 0; full_coord[1] = 0; full_coord[2] = 0; full_coord[3] = 0; full_coord[4] = 0;
    long output_coord[4];
    long remaining = output_index;
    output_coord[3] = remaining % u.output_extents[3]; remaining /= u.output_extents[3];
    output_coord[2] = remaining % u.output_extents[2]; remaining /= u.output_extents[2];
    output_coord[1] = remaining % u.output_extents[1]; remaining /= u.output_extents[1];
    output_coord[0] = remaining % u.output_extents[0]; remaining /= u.output_extents[0];
    full_coord[0] = output_coord[0]; full_coord[1] = output_coord[1]; full_coord[2] = output_coord[2]; full_coord[3] = output_coord[3];
    float accumulator;
    bool seeded;
    if (lane == 0u) { accumulator = 0.0f; seeded = true; } else { accumulator = 0.0f; seeded = true; }
    long stride0 = u.operand_strides[0][4];
    long off0 = u.operand_base[0];
    off0 += full_coord[0] * u.operand_strides[0][0];
    off0 += full_coord[1] * u.operand_strides[0][1];
    off0 += full_coord[2] * u.operand_strides[0][2];
    off0 += full_coord[3] * u.operand_strides[0][3];
    off0 += (long)lane * stride0;
    int walk0 = (int)off0;
    int advance0 = (int)(stride0 * 32);
    long stride1 = u.operand_strides[1][4];
    long off1 = u.operand_base[1];
    off1 += full_coord[0] * u.operand_strides[1][0];
    off1 += full_coord[1] * u.operand_strides[1][1];
    off1 += full_coord[2] * u.operand_strides[1][2];
    off1 += full_coord[3] * u.operand_strides[1][3];
    off1 += (long)lane * stride1;
    int walk1 = (int)off1;
    int advance1 = (int)(stride1 * 32);
    for (int r = (int)lane; r < (int)u.reduction_total; r += 32) {
        float scratch[2];
        scratch[0] = in0[walk0];
        scratch[1] = in1[walk1];
        float value = scratch[0] * scratch[1];
        accumulator = seeded ? (accumulator + value) : value;
        seeded = true;
        walk0 += advance0;
        walk1 += advance1;
    }
    if (output_index == 16) { diag_final[lane] = accumulator; }
    float reduced = simd_sum(accumulator);
    if (lane == 0u) {
        long out_offset = u.out_base;
        out_offset += full_coord[0] * u.out_strides[0];
        out_offset += full_coord[1] * u.out_strides[1];
        out_offset += full_coord[2] * u.out_strides[2];
        out_offset += full_coord[3] * u.out_strides[3];
        out_offset += full_coord[4] * u.out_strides[4];
        float epi_scratch[4];
        long epi_off0 = u.epilogue_operand_base[0];
        epi_off0 += output_coord[0] * u.epilogue_operand_strides[0][0];
        epi_off0 += output_coord[1] * u.epilogue_operand_strides[0][1];
        epi_off0 += output_coord[2] * u.epilogue_operand_strides[0][2];
        epi_off0 += output_coord[3] * u.epilogue_operand_strides[0][3];
        epi_scratch[0] = epi0[epi_off0];
        long epi_off1 = u.epilogue_operand_base[1];
        epi_off1 += output_coord[0] * u.epilogue_operand_strides[1][0];
        epi_off1 += output_coord[1] * u.epilogue_operand_strides[1][1];
        epi_off1 += output_coord[2] * u.epilogue_operand_strides[1][2];
        epi_off1 += output_coord[3] * u.epilogue_operand_strides[1][3];
        epi_scratch[1] = epi1[epi_off1];
        long epi_off2 = u.epilogue_operand_base[2];
        epi_off2 += output_coord[0] * u.epilogue_operand_strides[2][0];
        epi_off2 += output_coord[1] * u.epilogue_operand_strides[2][1];
        epi_off2 += output_coord[2] * u.epilogue_operand_strides[2][2];
        epi_off2 += output_coord[3] * u.epilogue_operand_strides[2][3];
        epi_scratch[2] = epi2[epi_off2];
        epi_scratch[3] = reduced;
        float epi_step0 = epi_scratch[3];
        float epi_step1 = (epi_step0 + epi_scratch[2]);
        float epi_step2 = ((epi_scratch[0] != 0.0f) ? epi_scratch[1] : epi_step1);
        out[out_offset] = epi_step2;
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

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn pack_uniforms_node139(output_total: i64, reduction_total: i64, output_extents: [i64; 4], operand_strides: [[i64; 5]; 2], out_strides: [i64; 5]) -> Vec<u8> {
    let mut fields: Vec<i64> = Vec::new();
    fields.push(output_total);
    fields.push(reduction_total);
    fields.extend_from_slice(&output_extents);
    fields.push(reduction_total);
    fields.extend_from_slice(&[0, 0]);
    fields.extend_from_slice(&operand_strides[0]);
    fields.extend_from_slice(&operand_strides[1]);
    fields.push(0);
    fields.extend_from_slice(&out_strides);
    i64_bytes(&fields)
}

fn pack_uniforms_node142(output_total: i64, reduction_total: i64, output_extents: [i64; 4], operand_strides: [[i64; 5]; 2], out_strides: [i64; 5], epilogue_strides: [[i64; 4]; 3]) -> Vec<u8> {
    let mut bytes = pack_uniforms_node139(output_total, reduction_total, output_extents, operand_strides, out_strides);
    let mut tail: Vec<i64> = Vec::new();
    tail.extend_from_slice(&[0, 0, 0]);
    tail.extend_from_slice(&epilogue_strides[0]);
    tail.extend_from_slice(&epilogue_strides[1]);
    tail.extend_from_slice(&epilogue_strides[2]);
    bytes.extend_from_slice(&i64_bytes(&tail));
    bytes
}

fn dispatch(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[&ProtocolObject<dyn MTLBuffer>],
    grid_threads: usize,
    threadgroup_width: usize,
) {
    let command_buffer = queue.commandBuffer().expect("command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("compute encoder");
    encoder.setComputePipelineState(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        unsafe { encoder.setBuffer_offset_atIndex(Some(*buffer), 0, index) };
    }
    let grid = MTLSize { width: grid_threads, height: 1, depth: 1 };
    let threadgroup = MTLSize { width: threadgroup_width, height: 1, depth: 1 };
    encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let _ = device;
}

// production fused kernel body verbatim from `attn_fused_replay.rs`'s
// `DIAGNOSTIC_KERNEL` (7-key score stores, already gate-proven to
// reproduce production out0/out1), with ONE addition: `diag_partial_score`
// stores each lane's OWN `partial_score` for key=2, group=0, BEFORE
// `simd_broadcast_first(simd_sum(...))` -- one store, after the loop, no
// per-step stores inside it.
const FUSED_PARTIAL_SCORE_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Uniforms { long total_elements; };

kernel void fused_partial_score(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const float* in2 [[buffer(2)]], device const float* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const float* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], device float* diag_score [[buffer(11)]], device float* diag_partial_score [[buffer(12)]], uint gid [[thread_position_in_grid]]) {
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
        if (key == 2 && group == 0) { diag_partial_score[lane] = partial_score; }
        float score = simd_broadcast_first(simd_sum(partial_score)) * scale;
        if (group == 0 && lane == 0u) { diag_score[key] = score; }
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

// same body, `#pragma METAL fp contract(off)` prepended -- the toggle: if
// this alone repairs t=2 against the unfused reference, FMA contraction in
// the fused kernel's interleaved accumulator is the root cause.
const FUSED_CONTRACT_OFF_KERNEL: &str = r#"
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

struct Uniforms { long total_elements; };

kernel void fused_contract_off(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const float* in2 [[buffer(2)]], device const float* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const float* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], device float* diag_score [[buffer(11)]], device float* diag_partial_score [[buffer(12)]], uint gid [[thread_position_in_grid]]) {
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
        if (key == 2 && group == 0) { diag_partial_score[lane] = partial_score; }
        float score = simd_broadcast_first(simd_sum(partial_score)) * scale;
        if (group == 0 && lane == 0u) { diag_score[key] = score; }
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

// round 7: COMBINED toggle -- production-shaped fused kernel with BOTH (a)
// split partial_even/partial_odd accumulators in node 142/139's own per-lane
// order (instead of one interleaved partial_score) and (b) `fp contract(off)`,
// then `score = (simd_sum(partial_even) + simd_sum(partial_odd)) * scale` in
// node 142's own epilogue order (reduced_even + node139's separately-reduced
// odd sum). Captures per-lane partial_even/partial_odd at key=2,group=0 for
// the round-6 node142/node139 parity gate, plus all 7 scores.
const COMBINED_SPLIT_CONTRACT_OFF_KERNEL: &str = r#"
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

struct Uniforms { long total_elements; };

kernel void combined_split_contract_off(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const float* in2 [[buffer(2)]], device const float* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const float* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], device float* diag_score [[buffer(11)]], device float* diag_partial_even [[buffer(12)]], device float* diag_partial_odd [[buffer(13)]], uint gid [[thread_position_in_grid]]) {
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
        if (key == 2 && group == 0) { diag_partial_even[lane] = partial_even; diag_partial_odd[lane] = partial_odd; }
        float score = (simd_sum(partial_even) + simd_sum(partial_odd)) * scale;
        if (group == 0 && lane == 0u) { diag_score[key] = score; }
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

// round 7 downstream walk stage (i)-(iii): same COMBINED body, plus a
// diagnostic capture of the post-chunk-merge `maximum`/`sum`/`weighted[0]`
// for group=0 (one store, at the same point the kernel already commits to
// them before dividing) -- gate target is node152 (max), node157+node158
// (sum), node162[0]+node164[0] (weighted[0], pre-normalize).
const COMBINED_MERGE_DIAG_KERNEL: &str = r#"
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

struct Uniforms { long total_elements; };

kernel void combined_merge_diag(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const float* in2 [[buffer(2)]], device const float* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const float* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], device float* diag_merge [[buffer(11)]], uint gid [[thread_position_in_grid]]) {
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
        float score = (simd_sum(partial_even) + simd_sum(partial_odd)) * scale;
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
        if (lane == 0u && group == 0L) { diag_merge[0] = merged_max; diag_merge[1] = merged_sum; diag_merge[2] = weighted[0]; }
        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) { long local_dimension = dimension / 32L; out[query_index * head_dim + dimension] = (float)(sum == 0.0f ? 0.0f : weighted[local_dimension] / sum); }
    }
}
"#;

// unmodified production fused kernel (verbatim, for the out0/out1 gate).
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

const REFERENCE_SCORE_BITS: [u32; 7] = [
    1092639986, 1090548168, 1077815366, 1087431027, 1088302311, 1089938621, 1086792084,
];

// round 7: unfused-native node=166 group-0 output vector (256 f32, bit
// pattern), captured by `payload_r2_safe2.txt:59` -- the downstream-walk and
// combined-toggle output gate target.
const NODE166_UNFUSED_REFERENCE: [u32; 256] = [
    1033978484, 3196771253, 3188208517, 1025766214, 3189800086, 3200142973, 3192833225, 3185079637,
    3183158661, 1044943177, 3187683287, 3188621327, 1044079111, 1013550076, 1036394317, 1031294342,
    1036790357, 3217442740, 3210971381, 1049343999, 1042414492, 1031710242, 1078339495, 1053356841,
    3175169511, 1033244130, 3193225699, 3185653119, 1032990157, 3189528308, 3184272268, 3159405523,
    1055249291, 3218745753, 1057944834, 1051644316, 3208950836, 3231414428, 3180719139, 1062737434,
    1027137850, 1056284327, 1036412802, 1039894875, 1059944491, 1048436350, 1040988014, 1064239596,
    1060469090, 1053696873, 1069943225, 1072784056, 1038060256, 1064575999, 1073075812, 1047930602,
    1061674940, 3191368848, 1052313144, 3202290924, 1051126265, 3164437833, 3212091574, 1027475175,
    1060498341, 3206016514, 3146843152, 1042971405, 3205854867, 3215941342, 3199194211, 1048357973,
    3207234346, 1061542321, 3186128172, 1042262694, 1050877354, 3199011532, 3171827809, 1031927485,
    1067958399, 3188412993, 1051551577, 1038854806, 1057073597, 1061191454, 1061847194, 1055529462,
    1052734074, 1011350356, 3191626762, 3214585402, 1056158465, 3207069070, 1058276590, 1009669778,
    1039713827, 3199751940, 3155018957, 1032821605, 3192264268, 3167474018, 3191099403, 3161543912,
    3177711650, 1048952652, 3182003064, 3188356550, 1043984862, 996532748, 1035986355, 1030614263,
    1044389668, 3217249793, 3211765299, 1052124691, 1036019503, 1044449645, 1078821129, 1049941150,
    3166463310, 1043630713, 3197048480, 3178878469, 1036700951, 3194391715, 3190972948, 1015249669,
    1015296176, 3190768809, 3176886299, 1018968995, 3180400728, 1066294617, 3190389193, 3191791458,
    981588487, 1043151726, 3183711153, 3192927163, 1032291597, 1018904407, 1038737510, 3165506689,
    3172180968, 3221404050, 3216731078, 1045715014, 1011885276, 1017532852, 1081480178, 1046903198,
    3190048739, 1049103072, 3200123051, 1042405632, 3154873851, 3192325290, 3193775488, 1020687005,
    1058456356, 3195651113, 1036160621, 1052372253, 3200599798, 1068660215, 3205281989, 1049053304,
    3204634191, 1062607004, 3179150328, 1042471639, 1048713111, 3195557771, 3176447956, 3175573516,
    1066067969, 3209587180, 3203554413, 3189936230, 1058933517, 1057401576, 1069717198, 1049706909,
    1040616220, 3188449190, 3179152347, 3210912164, 1048933427, 3208228141, 1052833573, 1045962653,
    1003293101, 3192840409, 3164028097, 1030852254, 3183615678, 1066084888, 3193218961, 3175522880,
    3160672151, 1046577578, 3184519604, 3191888651, 1041806026, 3156245347, 1034550667, 3176784990,
    1033846782, 3220315926, 3215627172, 1043678559, 1034263489, 1031885480, 1080424273, 1049530974,
    3190645931, 1044349138, 3200026808, 1026900567, 1025771817, 3195990806, 3190307443, 1026924348,
    1063113772, 3208522320, 1045341409, 1036785429, 3209002152, 1070472655, 3203347198, 1056808872,
    3214281332, 1065911077, 3194194857, 1046403749, 1051870420, 3208747979, 3190168838, 3177185220,
    1074314030, 3185783878, 1058931975, 3199900823, 1058901150, 1065806583, 1050867416, 1055748218,
    1055463958, 1023755593, 3205167391, 3221544724, 1059404759, 3213771957, 1066277791, 1032371426,
];

// production-STRUCTURED (runtime-derived qbase/kbase, `cached ? :`
// ternary, threadgroup memory, barrier, 768-thread/24-simdgroup dispatch)
// separate partial_even accumulation, byte-identical to round 5's
// `ISOLATED_LANE_DIAG_KERNEL` except it stores ONLY the final
// `partial_even` per lane (single store, after the loop, before
// `simd_sum`) -- gate target is the round-5 captured value 0x3b7ac2a1 at
// lane=1.
const ISOLATED_LANE_DIAG_FINAL_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Uniforms { long total_elements; };

kernel void isolated_lane_diag_final(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const float* in2 [[buffer(2)]], device const float* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const float* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], device float* diag_final [[buffer(11)]], uint gid [[thread_position_in_grid]]) {
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
        if (group == 0) { diag_final[lane] = partial_even; lane_even[lane] = partial_even; lane_odd[lane] = partial_odd; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float simd_even = simd_sum(partial_even);
        (void)simd_even;
    }
}
"#;

// same body, `#pragma METAL fp contract(off)` prepended.
const ISOLATED_LANE_DIAG_FINAL_CONTRACT_OFF_KERNEL: &str = r#"
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

struct Uniforms { long total_elements; };

kernel void isolated_lane_diag_final_contract_off(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const float* in2 [[buffer(2)]], device const float* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const float* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], device float* diag_final [[buffer(11)]], uint gid [[thread_position_in_grid]]) {
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
        if (group == 0) { diag_final[lane] = partial_even; lane_even[lane] = partial_even; lane_odd[lane] = partial_odd; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float simd_even = simd_sum(partial_even);
        (void)simd_even;
    }
}
"#;

// exhaustive per-step contraction pattern search: tries every 2^n
// combination of separate-vs-fma per step and returns the (pattern, value)
// pairs that reproduce `target` bit-exactly. `pairs` is (q, k) in the
// exact order the kernel accumulates them.
fn host_replay(pairs: &[(f32, f32)], fma_mask: u32) -> f32 {
    let mut accumulator = 0.0_f32;
    for (index, (q, k)) in pairs.iter().enumerate() {
        if (fma_mask >> index) & 1 == 1 {
            accumulator = q.mul_add(*k, accumulator);
        } else {
            accumulator += q * k;
        }
    }
    accumulator
}

fn find_matching_patterns(pairs: &[(f32, f32)], target_bits: u32) -> Vec<u32> {
    let combinations: u32 = 1 << pairs.len();
    (0..combinations).filter(|&mask| host_replay(pairs, mask).to_bits() == target_bits).collect()
}

fn pattern_string(mask: u32, steps: usize) -> String {
    (0..steps).map(|index| if (mask >> index) & 1 == 1 { 'F' } else { 'S' }).collect()
}

pub fn run() {
    let manifest_path = std::env::var("ATTN_FIXTURE_MANIFEST")
        .expect("ATTN_FIXTURE_MANIFEST must point at the round-4 manifest_safe_nodekernels.txt");
    let manifest = std::fs::read_to_string(&manifest_path).expect("read fixture manifest");

    let q_even = parse_leaf(&manifest, "q_even");
    let q_odd = parse_leaf(&manifest, "q_odd");
    let k_even_cache = parse_leaf(&manifest, "k_even_cache");
    let k_odd_cache = parse_leaf(&manifest, "k_odd_cache");
    let new_k_even = parse_leaf(&manifest, "new_k_even");
    let new_k_odd = parse_leaf(&manifest, "new_k_odd");
    let v_cache = parse_leaf(&manifest, "v_cache");
    let v_new = parse_leaf(&manifest, "v_new");

    let device = MTLCreateSystemDefaultDevice().expect("system default Metal device");
    let queue = device.newCommandQueue().expect("command queue");

    let output_extents: [i64; 4] = [1, 32, 1, 8];
    let output_total: i64 = 256;
    let reduction_total: i64 = 128;
    let k_strides: [i64; 5] = [0, 128, 128, 0, 1];
    let q_strides: [i64; 5] = [1024, 0, 1024, 128, 1];
    let out_strides: [i64; 5] = [256, 8, 8, 1, 0];
    let epi_strides: [[i64; 4]; 3] = [[32, 1, 0, 0], [0, 0, 0, 0], [256, 8, 8, 1]];

    // === GATE 1: node139/node142 FINAL-only kernels reproduce production bits ===
    let in0_odd = shared_buffer(&device, &f32_bytes(&k_odd_cache));
    let in1_odd = shared_buffer(&device, &f32_bytes(&q_odd));
    let out139 = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 256]));
    let uniforms139 = shared_buffer(&device, &pack_uniforms_node139(output_total, reduction_total, output_extents, [k_strides, q_strides], out_strides));
    let diag139_final = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 32]));
    let pipeline139 = compile(&device, NODE139_FINAL_KERNEL, "final_node139", MTLMathMode::Safe);
    dispatch(&device, &queue, &pipeline139, &[&in0_odd, &in1_odd, &out139, &uniforms139, &diag139_final], 8192, 32);
    let out139_values = read_f32_buffer(&out139, 256);
    let diag139_final_values = read_f32_buffer(&diag139_final, 32);
    const NODE139_REFERENCE: u32 = 0xbea5746d;
    println!(
        "attn_contraction_replay GATE node139_final output=0x{:08x} reference=0x{NODE139_REFERENCE:08x} match={}",
        out139_values[16].to_bits(), out139_values[16].to_bits() == NODE139_REFERENCE,
    );

    let in0_even = shared_buffer(&device, &f32_bytes(&k_even_cache));
    let in1_even = shared_buffer(&device, &f32_bytes(&q_even));
    let epi0_mask: Vec<f32> = {
        let mut mask = vec![1.0_f32; 32];
        mask[0] = 0.0; mask[1] = 0.0; mask[2] = 0.0;
        mask
    };
    let epi1_neg_inf = vec![f32::NEG_INFINITY];
    let epi0_buf = shared_buffer(&device, &f32_bytes(&epi0_mask));
    let epi1_buf = shared_buffer(&device, &f32_bytes(&epi1_neg_inf));
    let epi2_buf = shared_buffer(&device, &f32_bytes(&out139_values));
    let out142 = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 256]));
    let uniforms142 = shared_buffer(&device, &pack_uniforms_node142(output_total, reduction_total, output_extents, [k_strides, q_strides], out_strides, epi_strides));
    let diag142_final = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 32]));
    let pipeline142 = compile(&device, NODE142_FINAL_KERNEL, "final_node142", MTLMathMode::Safe);
    dispatch(&device, &queue, &pipeline142, &[&in0_even, &in1_even, &epi0_buf, &epi1_buf, &epi2_buf, &out142, &uniforms142, &diag142_final], 8192, 32);
    let out142_values = read_f32_buffer(&out142, 256);
    let diag142_final_values = read_f32_buffer(&diag142_final, 32);
    const NODE142_REFERENCE: u32 = 0x403e2846;
    println!(
        "attn_contraction_replay GATE node142_final output=0x{:08x} reference=0x{NODE142_REFERENCE:08x} match={}",
        out142_values[16].to_bits(), out142_values[16].to_bits() == NODE142_REFERENCE,
    );

    // converse control: contract(off) on the REAL node142 kernel
    let diag142_off_final = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 32]));
    let out142_off = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 256]));
    let pipeline142_off = compile(&device, NODE142_FINAL_KERNEL_CONTRACT_OFF, "final_node142_contract_off", MTLMathMode::Safe);
    dispatch(&device, &queue, &pipeline142_off, &[&in0_even, &in1_even, &epi0_buf, &epi1_buf, &epi2_buf, &out142_off, &uniforms142, &diag142_off_final], 8192, 32);
    let out142_off_values = read_f32_buffer(&out142_off, 256);
    println!(
        "attn_contraction_replay CONTROL node142_contract_off output=0x{:08x} production_reference=0x{NODE142_REFERENCE:08x} changed_from_production={}",
        out142_off_values[16].to_bits(), out142_off_values[16].to_bits() != NODE142_REFERENCE,
    );

    // === GATE 2: fused partial-score kernel reproduces production out0/out1 ===
    let cached_len = shared_buffer(&device, &f32_bytes(&[6.0_f32]));
    let uniforms_total = shared_buffer(&device, &24_i64.to_le_bytes());
    let in4 = shared_buffer(&device, &f32_bytes(&new_k_even));
    let in5 = shared_buffer(&device, &f32_bytes(&new_k_odd));
    let in6 = shared_buffer(&device, &f32_bytes(&v_cache));
    let in7 = shared_buffer(&device, &f32_bytes(&v_new));

    let production_pipeline = compile(&device, PRODUCTION_KERNEL, "omega_cached_attention_q1_c32_n1_h1_g8_d256_s3f800000_ln511_up0_cb", MTLMathMode::Safe);
    let production_out = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 2048]));
    dispatch(&device, &queue, &production_pipeline, &[&in1_even, &in1_odd, &in0_even, &in0_odd, &in4, &in5, &in6, &in7, &cached_len, &production_out, &uniforms_total], 768, 768);
    let production_out_values = read_f32_buffer(&production_out, 2048);
    const PRODUCTION_OUT0: u32 = 0x3da14274;
    const PRODUCTION_OUT1: u32 = 0x3e0b0e8c;
    println!(
        "attn_contraction_replay GATE production out0=0x{:08x} ref=0x{PRODUCTION_OUT0:08x} match={} out1=0x{:08x} ref=0x{PRODUCTION_OUT1:08x} match={}",
        production_out_values[0].to_bits(), production_out_values[0].to_bits() == PRODUCTION_OUT0,
        production_out_values[1].to_bits(), production_out_values[1].to_bits() == PRODUCTION_OUT1,
    );

    let partial_score_pipeline = compile(&device, FUSED_PARTIAL_SCORE_KERNEL, "fused_partial_score", MTLMathMode::Safe);
    let partial_score_out = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 2048]));
    let partial_score_diag_score = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 16]));
    let partial_score_diag = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 32]));
    dispatch(
        &device, &queue, &partial_score_pipeline,
        &[&in1_even, &in1_odd, &in0_even, &in0_odd, &in4, &in5, &in6, &in7, &cached_len, &partial_score_out, &uniforms_total, &partial_score_diag_score, &partial_score_diag],
        768, 768,
    );
    let partial_score_out_values = read_f32_buffer(&partial_score_out, 2048);
    let partial_score_diag_score_values = read_f32_buffer(&partial_score_diag_score, 16);
    let fused_partial_score_values = read_f32_buffer(&partial_score_diag, 32);
    println!(
        "attn_contraction_replay GATE fused_partial_score_kernel out0=0x{:08x} ref=0x{PRODUCTION_OUT0:08x} match={} out1=0x{:08x} ref=0x{PRODUCTION_OUT1:08x} match={}",
        partial_score_out_values[0].to_bits(), partial_score_out_values[0].to_bits() == PRODUCTION_OUT0,
        partial_score_out_values[1].to_bits(), partial_score_out_values[1].to_bits() == PRODUCTION_OUT1,
    );
    for key in 0..7usize {
        println!(
            "attn_contraction_replay GATE fused_partial_score_kernel score t={key} bits=0x{:08x} ref=0x{:08x} match={}",
            partial_score_diag_score_values[key].to_bits(), REFERENCE_SCORE_BITS[key],
            partial_score_diag_score_values[key].to_bits() == REFERENCE_SCORE_BITS[key],
        );
    }

    // === host-side per-lane contraction-pattern search ===
    println!("attn_contraction_replay lane node139_pattern node142_pattern fused_pattern node139_all_sep node139_all_fma node142_all_sep node142_all_fma fused_all_sep fused_all_fma");
    let mut first_divergence: Option<(usize, usize)> = None;
    let mut fused_all_sep_by_lane: Vec<bool> = Vec::with_capacity(32);
    for lane in 0..32usize {
        let odd_pairs: Vec<(f32, f32)> = (0..4).map(|r| {
            let pair = lane + 32 * r;
            (q_odd[pair], k_odd_cache[256 + pair])
        }).collect();
        let even_pairs: Vec<(f32, f32)> = (0..4).map(|r| {
            let pair = lane + 32 * r;
            (q_even[pair], k_even_cache[256 + pair])
        }).collect();
        let fused_pairs: Vec<(f32, f32)> = (0..4).flat_map(|r| {
            let pair = lane + 32 * r;
            vec![(q_even[pair], k_even_cache[256 + pair]), (q_odd[pair], k_odd_cache[256 + pair])]
        }).collect();

        let node139_target = diag139_final_values[lane].to_bits();
        let node142_target = diag142_final_values[lane].to_bits();
        let fused_target = fused_partial_score_values[lane].to_bits();

        let node139_patterns = find_matching_patterns(&odd_pairs, node139_target);
        let node142_patterns = find_matching_patterns(&even_pairs, node142_target);
        let fused_patterns = find_matching_patterns(&fused_pairs, fused_target);

        let node139_all_sep = host_replay(&odd_pairs, 0).to_bits() == node139_target;
        let node139_all_fma = host_replay(&odd_pairs, 0b1111).to_bits() == node139_target;
        let node142_all_sep = host_replay(&even_pairs, 0).to_bits() == node142_target;
        let node142_all_fma = host_replay(&even_pairs, 0b1111).to_bits() == node142_target;
        let fused_all_sep = host_replay(&fused_pairs, 0).to_bits() == fused_target;
        let fused_all_fma = host_replay(&fused_pairs, 0b1111_1111).to_bits() == fused_target;

        println!(
            "attn_contraction_replay lane={lane} node139_patterns={:?} node142_patterns={:?} fused_patterns={:?} node139_all_sep={node139_all_sep} node139_all_fma={node139_all_fma} node142_all_sep={node142_all_sep} node142_all_fma={node142_all_fma} fused_all_sep={fused_all_sep} fused_all_fma={fused_all_fma}",
            node139_patterns.iter().map(|mask| pattern_string(*mask, 4)).collect::<Vec<_>>(),
            node142_patterns.iter().map(|mask| pattern_string(*mask, 4)).collect::<Vec<_>>(),
            fused_patterns.iter().map(|mask| pattern_string(*mask, 8)).collect::<Vec<_>>(),
        );
        fused_all_sep_by_lane.push(fused_all_sep);

        if lane == 1 {
            // owner's independent claim: real node142 (0x3b7ac2a0) is
            // reachable only via SEPARATE rounding at r=3; the round-5
            // LANE_DIAG/ISOLATED_LANE_DIAG production-shaped
            // separate-partial_even value (0x3b7ac2a1) is reachable only
            // via FMA at r=3. Verify against the SAME even_pairs both ways.
            let isolated_target: u32 = 0x3b7ac2a1;
            let node142_patterns_lane1 = find_matching_patterns(&even_pairs, node142_target);
            let isolated_patterns_lane1 = find_matching_patterns(&even_pairs, isolated_target);
            println!(
                "attn_contraction_replay LANE1_CLAIM_CHECK node142_target=0x{node142_target:08x} node142_patterns={:?} isolated_target=0x{isolated_target:08x} isolated_patterns={:?}",
                node142_patterns_lane1.iter().map(|mask| pattern_string(*mask, 4)).collect::<Vec<_>>(),
                isolated_patterns_lane1.iter().map(|mask| pattern_string(*mask, 4)).collect::<Vec<_>>(),
            );
            let r3_all_node142_separate = node142_patterns_lane1.iter().all(|mask| (mask >> 3) & 1 == 0);
            let r3_all_isolated_fma = isolated_patterns_lane1.iter().all(|mask| (mask >> 3) & 1 == 1);
            println!(
                "attn_contraction_replay LANE1_CLAIM_CHECK r3_all_node142_separate={r3_all_node142_separate} r3_all_isolated_fma={r3_all_isolated_fma} claim_holds={}",
                r3_all_node142_separate && r3_all_isolated_fma,
            );
        }

        if lane == 25 {
            let (q_step1, k_step1) = even_pairs[1];
            let acc_after_r0 = host_replay(&even_pairs[..1], 0);
            let separate_after_r1 = acc_after_r0 + q_step1 * k_step1;
            let fma_after_r1 = q_step1.mul_add(k_step1, acc_after_r0);
            println!(
                "attn_contraction_replay LANE25_R1_DETAIL q=0x{:08x} k=0x{:08x} acc_before=0x{:08x} separate_after=0x{:08x} fma_after=0x{:08x} separate_eq_fma={}",
                q_step1.to_bits(), k_step1.to_bits(), acc_after_r0.to_bits(),
                separate_after_r1.to_bits(), fma_after_r1.to_bits(),
                separate_after_r1.to_bits() == fma_after_r1.to_bits(),
            );
        }

        // "distinguished at r" means EVERY (node142_pattern, fused_pattern)
        // pair in the full compatible cross-product disagrees on the bit at
        // step r -- i.e. no compatible sequence pair on either side agrees.
        // If the sets are empty (no host form reproduces the kernel's own
        // bits at all -- pure cancellation, no single/no double rounding
        // sequence over 4/8 steps lands there) the step is reported as
        // UNDETERMINED, not distinguished.
        if first_divergence.is_none() {
            for r in 0..4usize {
                if node142_patterns.is_empty() || fused_patterns.is_empty() {
                    println!("attn_contraction_replay lane={lane} r={r} UNDETERMINED (empty compatible set: node142={} fused={})", node142_patterns.len(), fused_patterns.len());
                    continue;
                }
                let any_agree = node142_patterns.iter().any(|node142_mask| {
                    let node142_bit = (node142_mask >> r) & 1;
                    fused_patterns.iter().any(|fused_mask| ((fused_mask >> (r * 2)) & 1) == node142_bit)
                });
                if !any_agree {
                    first_divergence = Some((lane, r));
                    println!("attn_contraction_replay lane={lane} r={r} DISTINGUISHED (no compatible node142/fused pair agrees on this step's form)");
                    break;
                }
            }
        }
    }
    match first_divergence {
        Some((lane, r)) => println!("attn_contraction_replay FIRST_DIVERGENT_OP lane={lane} r={r} (every compatible node142 pattern disagrees with every compatible fused-even bit at this step)"),
        None => println!("attn_contraction_replay FIRST_DIVERGENT_OP none -- not distinguished by the evidence (some compatible sequence pair agrees at every step checked)"),
    }

    // === PREDICTION, written before the toggle runs ===
    // derived from `fused_all_sep_by_lane` (from the compatible-set search
    // above): a lane whose real `partial_score` is ALREADY reproducible by
    // the fully-separate (no-fma) host form is predicted to be UNCHANGED
    // by `fp contract(off)` (it was already behaving as separate); a lane
    // where `fused_all_sep=false` is predicted to CHANGE. Because the fused
    // kernel accumulates even/odd interleaved into ONE running accumulator
    // (a different association from node142's even-only + node139's
    // odd-only accumulators added at the epilogue), matching node142's own
    // FORM at every step is NOT predicted to make t=2 land exactly on the
    // unfused reference 0x403e2846 -- only that it moves.
    let predicted_changed_lanes: Vec<usize> = fused_all_sep_by_lane.iter().enumerate().filter(|(_, all_sep)| !**all_sep).map(|(lane, _)| lane).collect();
    println!(
        "attn_contraction_replay PREDICTION contract_off will change partial_score for {} of 32 lanes at key=2,group=0: {:?}. Predicted score t=2 changes from 0x403e2849; NOT predicted to land exactly on unfused_reference 0x403e2846 (interleaved association differs from node142+node139's split association).",
        predicted_changed_lanes.len(), predicted_changed_lanes,
    );

    // === toggle: fused kernel with fp contract(off) ===
    let toggle_pipeline = compile(&device, FUSED_CONTRACT_OFF_KERNEL, "fused_contract_off", MTLMathMode::Safe);
    let toggle_out = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 2048]));
    let toggle_diag = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 16]));
    let toggle_diag_partial = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 32]));
    dispatch(&device, &queue, &toggle_pipeline, &[&in1_even, &in1_odd, &in0_even, &in0_odd, &in4, &in5, &in6, &in7, &cached_len, &toggle_out, &uniforms_total, &toggle_diag, &toggle_diag_partial], 768, 768);
    let toggle_out_values = read_f32_buffer(&toggle_out, 2048);
    let toggle_diag_values = read_f32_buffer(&toggle_diag, 16);
    let toggle_diag_partial_values = read_f32_buffer(&toggle_diag_partial, 32);
    let hash: u64 = toggle_out_values[..256].iter().fold(0xcbf29ce484222325_u64, |accumulator, value| {
        (accumulator ^ u64::from(value.to_bits())).wrapping_mul(0x100000001b3)
    });
    println!(
        "attn_contraction_replay TOGGLE contract_off out0=0x{:08x} (was 0x{:08x}, round4_regression_expected=0x3da14273) out1=0x{:08x} (was 0x{:08x}, round4_regression_expected=0x3e0b0e8e) group0_hash=0x{hash:016x}",
        toggle_out_values[0].to_bits(), production_out_values[0].to_bits(),
        toggle_out_values[1].to_bits(), production_out_values[1].to_bits(),
    );
    println!(
        "attn_contraction_replay TOGGLE out0_matches_round4_regression={} out1_matches_round4_regression={}",
        toggle_out_values[0].to_bits() == 0x3da14273, toggle_out_values[1].to_bits() == 0x3e0b0e8e,
    );
    for key in 0..7usize {
        println!(
            "attn_contraction_replay TOGGLE score t={key} before=0x{:08x} after=0x{:08x} unfused_reference=0x{:08x} after_matches={} changed_from_before={}",
            partial_score_diag_score_values[key].to_bits(), toggle_diag_values[key].to_bits(), REFERENCE_SCORE_BITS[key],
            toggle_diag_values[key].to_bits() == REFERENCE_SCORE_BITS[key],
            toggle_diag_values[key].to_bits() != partial_score_diag_score_values[key].to_bits(),
        );
    }
    println!("attn_contraction_replay PREDICTION_VS_OBSERVATION lane before_bits after_bits changed predicted_changed prediction_correct");
    let mut prediction_hits = 0usize;
    for lane in 0..32usize {
        let before_bits = fused_partial_score_values[lane].to_bits();
        let after_bits = toggle_diag_partial_values[lane].to_bits();
        let observed_changed = before_bits != after_bits;
        let predicted_changed = !fused_all_sep_by_lane[lane];
        let prediction_correct = observed_changed == predicted_changed;
        if prediction_correct { prediction_hits += 1; }
        println!(
            "attn_contraction_replay PREDICTION_VS_OBSERVATION lane={lane} before=0x{before_bits:08x} after=0x{after_bits:08x} changed={observed_changed} predicted_changed={predicted_changed} prediction_correct={prediction_correct}",
        );
    }
    println!("attn_contraction_replay PREDICTION_SCORE {prediction_hits}/32 lanes matched the fused_all_sep-derived prediction");

    // === converse control already run above (node142 contract_off, printed as CONTROL) ===

    // === owner-directed check: production-STRUCTURED separate-partial_even
    // kernel, contract(on) gated against round-5's captured 0x3b7ac2a1,
    // then contract(off), final-accumulator-only, all 32 lanes ===
    let dummy4 = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 128]));
    let isolated_on_pipeline = compile(&device, ISOLATED_LANE_DIAG_FINAL_KERNEL, "isolated_lane_diag_final", MTLMathMode::Safe);
    let isolated_on_out = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 2048]));
    let isolated_on_diag = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 32]));
    dispatch(&device, &queue, &isolated_on_pipeline, &[&in1_even, &in1_odd, &in0_even, &in0_odd, &dummy4, &dummy4, &dummy4, &dummy4, &cached_len, &isolated_on_out, &uniforms_total, &isolated_on_diag], 768, 768);
    let isolated_on_values = read_f32_buffer(&isolated_on_diag, 32);
    const ISOLATED_LANE1_REFERENCE: u32 = 0x3b7ac2a1;
    println!(
        "attn_contraction_replay GATE isolated_lane_diag_final lane=1 bits=0x{:08x} round5_reference=0x{ISOLATED_LANE1_REFERENCE:08x} match={}",
        isolated_on_values[1].to_bits(), isolated_on_values[1].to_bits() == ISOLATED_LANE1_REFERENCE,
    );

    let isolated_off_pipeline = compile(&device, ISOLATED_LANE_DIAG_FINAL_CONTRACT_OFF_KERNEL, "isolated_lane_diag_final_contract_off", MTLMathMode::Safe);
    let isolated_off_out = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 2048]));
    let isolated_off_diag = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 32]));
    dispatch(&device, &queue, &isolated_off_pipeline, &[&in1_even, &in1_odd, &in0_even, &in0_odd, &dummy4, &dummy4, &dummy4, &dummy4, &cached_len, &isolated_off_out, &uniforms_total, &isolated_off_diag], 768, 768);
    let isolated_off_values = read_f32_buffer(&isolated_off_diag, 32);

    println!("attn_contraction_replay ISOLATED_TOGGLE lane on(contract_default) off(contract_off) node142_real predicted_lane1=0x3b7ac2a0 matches_node142");
    for lane in 0..32usize {
        let on_bits = isolated_on_values[lane].to_bits();
        let off_bits = isolated_off_values[lane].to_bits();
        let node142_bits = diag142_final_values[lane].to_bits();
        println!(
            "attn_contraction_replay ISOLATED_TOGGLE lane={lane} on=0x{on_bits:08x} off=0x{off_bits:08x} node142_real=0x{node142_bits:08x} off_matches_node142={} changed_by_toggle={}",
            off_bits == node142_bits, on_bits != off_bits,
        );
    }

    // === round 7: COMBINED toggle (split accumulators + contract(off),
    // production-shaped full softmax/merge/normalize body) ===
    let combined_pipeline = compile(&device, COMBINED_SPLIT_CONTRACT_OFF_KERNEL, "combined_split_contract_off", MTLMathMode::Safe);
    let combined_out = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 2048]));
    let combined_diag_score = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 16]));
    let combined_diag_even = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 32]));
    let combined_diag_odd = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 32]));
    dispatch(
        &device, &queue, &combined_pipeline,
        &[&in1_even, &in1_odd, &in0_even, &in0_odd, &in4, &in5, &in6, &in7, &cached_len, &combined_out, &uniforms_total, &combined_diag_score, &combined_diag_even, &combined_diag_odd],
        768, 768,
    );
    let combined_out_values = read_f32_buffer(&combined_out, 2048);
    let combined_diag_score_values = read_f32_buffer(&combined_diag_score, 16);
    let combined_diag_even_values = read_f32_buffer(&combined_diag_even, 32);
    let combined_diag_odd_values = read_f32_buffer(&combined_diag_odd, 32);

    println!("attn_contraction_replay COMBINED per_lane_even lane combined node142_real match");
    let mut even_matches = 0usize;
    for lane in 0..32usize {
        let combined_bits = combined_diag_even_values[lane].to_bits();
        let node142_bits = diag142_final_values[lane].to_bits();
        let matches = combined_bits == node142_bits;
        if matches { even_matches += 1; }
        println!(
            "attn_contraction_replay COMBINED per_lane_even lane={lane} combined=0x{combined_bits:08x} node142_real=0x{node142_bits:08x} match={matches}",
        );
    }
    println!("attn_contraction_replay COMBINED per_lane_even_score {even_matches}/32");

    println!("attn_contraction_replay COMBINED per_lane_odd lane combined node139_real match");
    let mut odd_matches = 0usize;
    for lane in 0..32usize {
        let combined_bits = combined_diag_odd_values[lane].to_bits();
        let node139_bits = diag139_final_values[lane].to_bits();
        let matches = combined_bits == node139_bits;
        if matches { odd_matches += 1; }
        println!(
            "attn_contraction_replay COMBINED per_lane_odd lane={lane} combined=0x{combined_bits:08x} node139_real=0x{node139_bits:08x} match={matches}",
        );
    }
    println!("attn_contraction_replay COMBINED per_lane_odd_score {odd_matches}/32");

    println!("attn_contraction_replay COMBINED score t combined unfused_reference match");
    let mut score_matches = 0usize;
    for key in 0..7usize {
        let combined_bits = combined_diag_score_values[key].to_bits();
        let matches = combined_bits == REFERENCE_SCORE_BITS[key];
        if matches { score_matches += 1; }
        println!(
            "attn_contraction_replay COMBINED score t={key} combined=0x{combined_bits:08x} unfused_reference=0x{:08x} match={matches}",
            REFERENCE_SCORE_BITS[key],
        );
    }
    println!("attn_contraction_replay COMBINED score_match {score_matches}/7");

    const PRODUCTION_OUT0_REF: u32 = 0x3da14274;
    const PRODUCTION_OUT1_REF: u32 = 0x3e0b0e8c;
    println!(
        "attn_contraction_replay COMBINED out0=0x{:08x} production_ref=0x{PRODUCTION_OUT0_REF:08x} out0_matches_production={} out1=0x{:08x} production_ref=0x{PRODUCTION_OUT1_REF:08x} out1_matches_production={}",
        combined_out_values[0].to_bits(), combined_out_values[0].to_bits() == PRODUCTION_OUT0_REF,
        combined_out_values[1].to_bits(), combined_out_values[1].to_bits() == PRODUCTION_OUT1_REF,
    );

    let combined_hash: u64 = combined_out_values[..256].iter().fold(0xcbf29ce484222325_u64, |accumulator, value| {
        (accumulator ^ u64::from(value.to_bits())).wrapping_mul(0x100000001b3)
    });
    let unfused_hash: u64 = NODE166_UNFUSED_REFERENCE.iter().fold(0xcbf29ce484222325_u64, |accumulator, bits| {
        (accumulator ^ u64::from(*bits)).wrapping_mul(0x100000001b3)
    });
    let first_diff = (0..256usize).find(|index| combined_out_values[*index].to_bits() != NODE166_UNFUSED_REFERENCE[*index]);
    println!(
        "attn_contraction_replay COMBINED group0_hash=0x{combined_hash:016x} unfused_node166_hash=0x{unfused_hash:016x} hashes_match={} first_diff_index={:?}",
        combined_hash == unfused_hash, first_diff,
    );
    if let Some(index) = first_diff {
        println!(
            "attn_contraction_replay COMBINED first_diff index={index} combined=0x{:08x} unfused_reference=0x{:08x}",
            combined_out_values[index].to_bits(), NODE166_UNFUSED_REFERENCE[index],
        );
    }

    // === round 7 downstream walk stage (i)-(iii): merged max/sum/weighted[0] ===
    let merge_pipeline = compile(&device, COMBINED_MERGE_DIAG_KERNEL, "combined_merge_diag", MTLMathMode::Safe);
    let merge_out = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 2048]));
    let merge_diag = shared_buffer(&device, &f32_bytes(&vec![0.0_f32; 3]));
    dispatch(&device, &queue, &merge_pipeline, &[&in1_even, &in1_odd, &in0_even, &in0_odd, &in4, &in5, &in6, &in7, &cached_len, &merge_out, &uniforms_total, &merge_diag], 768, 768);
    let merge_diag_values = read_f32_buffer(&merge_diag, 3);
    const NODE152_MAX_REFERENCE: u32 = 1092639986; // node=152 qg0_column[0], payload_r2_safe2.txt
    const NODE157_PLUS_158_SUM_BITS: (u32, u32) = (1067983221, 1018645065); // node=157, node=158
    const NODE162_PLUS_164_WEIGHTED0_BITS: (u32, u32) = (1039474757, 3161299426); // node=162[0], node=164[0]
    let merged_max_bits = merge_diag_values[0].to_bits();
    let merged_sum_bits = merge_diag_values[1].to_bits();
    let merged_weighted0_bits = merge_diag_values[2].to_bits();
    let node157 = f32::from_bits(NODE157_PLUS_158_SUM_BITS.0);
    let node158 = f32::from_bits(NODE157_PLUS_158_SUM_BITS.1);
    let unfused_sum = node157 + node158;
    let node162_0 = f32::from_bits(NODE162_PLUS_164_WEIGHTED0_BITS.0);
    let node164_0 = f32::from_bits(NODE162_PLUS_164_WEIGHTED0_BITS.1);
    let unfused_weighted0 = node162_0 + node164_0;
    println!(
        "attn_contraction_replay COMBINED_DOWNSTREAM stage=max combined=0x{merged_max_bits:08x} unfused_node152=0x{NODE152_MAX_REFERENCE:08x} match={}",
        merged_max_bits == NODE152_MAX_REFERENCE,
    );
    println!(
        "attn_contraction_replay COMBINED_DOWNSTREAM stage=sum combined=0x{merged_sum_bits:08x} unfused_node157_plus_node158=0x{:08x} (node157=0x{:08x} node158=0x{:08x}) match={}",
        unfused_sum.to_bits(), NODE157_PLUS_158_SUM_BITS.0, NODE157_PLUS_158_SUM_BITS.1,
        merged_sum_bits == unfused_sum.to_bits(),
    );
    println!(
        "attn_contraction_replay COMBINED_DOWNSTREAM stage=weighted0 combined=0x{merged_weighted0_bits:08x} unfused_node162_plus_node164=0x{:08x} (node162[0]=0x{:08x} node164[0]=0x{:08x}) match={}",
        unfused_weighted0.to_bits(), NODE162_PLUS_164_WEIGHTED0_BITS.0, NODE162_PLUS_164_WEIGHTED0_BITS.1,
        merged_weighted0_bits == unfused_weighted0.to_bits(),
    );
    let divide_form = merge_diag_values[2] / (merge_diag_values[1]);
    let recip_mul_form = merge_diag_values[2] * (1.0_f32 / merge_diag_values[1]);
    println!(
        "attn_contraction_replay COMBINED_DOWNSTREAM stage=normalize(own_inputs) divide_form=0x{:08x} recip_mul_form=0x{:08x} unfused_node166_0=0x{:08x} divide_matches={} recip_mul_matches={}",
        divide_form.to_bits(), recip_mul_form.to_bits(), NODE166_UNFUSED_REFERENCE[0],
        divide_form.to_bits() == NODE166_UNFUSED_REFERENCE[0], recip_mul_form.to_bits() == NODE166_UNFUSED_REFERENCE[0],
    );
    // isolate the final op's rounding form using the UNFUSED reference's own
    // weighted0/sum (not ours) -- tests divide-vs-reciprocal-multiply alone,
    // uncontaminated by any upstream schedule mismatch.
    let unfused_divide_form = unfused_weighted0 / unfused_sum;
    let unfused_recip_mul_form = unfused_weighted0 * (1.0_f32 / unfused_sum);
    println!(
        "attn_contraction_replay COMBINED_DOWNSTREAM stage=normalize(unfused_inputs) divide_form=0x{:08x} recip_mul_form=0x{:08x} unfused_node166_0=0x{:08x} divide_matches={} recip_mul_matches={}",
        unfused_divide_form.to_bits(), unfused_recip_mul_form.to_bits(), NODE166_UNFUSED_REFERENCE[0],
        unfused_divide_form.to_bits() == NODE166_UNFUSED_REFERENCE[0], unfused_recip_mul_form.to_bits() == NODE166_UNFUSED_REFERENCE[0],
    );
}

}
