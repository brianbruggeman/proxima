//! Level-5 stratification of `cpu::dot_q4k_q8k_block_neon_dotprod`: where
//! do the ~250 ms of `own_chunk_ms` actually go INSIDE one `Q4_K`
//! super-block?
//!
//! Every prior attempt on this gap named a mechanism (cold cache, page
//! faults, instruction volume, wake storms, spin waste) and tested it
//! alone. This bench does not name a mechanism. It reconstructs the real
//! super-block kernel locally, byte for byte, against real packed
//! checkpoint bytes, and removes ONE phase per arm. Each arm's delta from
//! the full kernel IS that phase's cost; the arms are required to sum back
//! to the full kernel and the residual is reported rather than normalized
//! away.
//!
//! The reconstruction is only admissible if it reproduces the shipped
//! kernel's own rate: arm `real_dot_q4k_q8k` calls
//! `proxima_tensor::cpu::dot_q4k_q8k` over the identical bytes in the
//! identical order, and `local_full` must land on it. A reconstruction
//! that does not match is measuring something else and its ablations mean
//! nothing.
//!
//! Every arm is `#[inline(never)]` so its instruction count can be read
//! straight out of the built binary (`objdump -d`, symbol `q4k_arm_*`).
//! That costs one call/return per super-block in EVERY arm, so the bias is
//! uniform and the arm-to-arm deltas -- the thing being measured -- are
//! unaffected.
//!
//! Two cache configurations, because the real forward streams distinct
//! weight bytes exactly once while a single-buffer microbench re-reads a
//! resident one:
//! - `warm`: one tensor (`blk.0.attn_q.weight`), re-read every pass.
//! - `cold`: the same tensor from all 32 blocks, ~300 MB, one pass over
//!   all of them so no line survives to the next visit.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

#[cfg(target_arch = "aarch64")]
use std::fs::File;
#[cfg(target_arch = "aarch64")]
use std::hint::black_box;
#[cfg(target_arch = "aarch64")]
use std::io::{Read, Seek, SeekFrom};
#[cfg(target_arch = "aarch64")]
use std::path::Path;
#[cfg(target_arch = "aarch64")]
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use proxima_gguf::parser::{GgufEvent, GgufParser};
#[cfg(target_arch = "aarch64")]
use proxima_gguf::pipe::ParsedGguf;
#[cfg(target_arch = "aarch64")]
use proxima_gguf::tensor::TensorInfo;
#[cfg(target_arch = "aarch64")]
use proxima_tensor::cpu::{dot_q4k_q8k, quantize_row_q8k};
#[cfg(target_arch = "aarch64")]
use proxima_tensor::test_support::Lcg;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    int8x16_t, int32x4_t, uint8x16_t, vaddq_s32, vaddq_u8, vaddvq_s32, vandq_u8, vdupq_n_s32,
    vdupq_n_u8, vget_high_s16, vget_low_s16, vld1_u32, vld1q_s8, vld1q_s16, vld1q_u8, vmovl_u8,
    vmull_s16, vpaddq_s16, vreinterpret_u8_u32, vreinterpretq_s8_u8, vreinterpretq_s16_u16,
    vshrq_n_u8,
};

/// real GGUF checkpoint path, overridable per-operator via
/// `PROXIMA_BENCH_GGUF_PATH` — the hardcoded default only ever resolved on
/// one machine, which made this bench unrunnable anywhere else.
#[cfg(target_arch = "aarch64")]
fn gguf_path() -> String {
    std::env::var("PROXIMA_BENCH_GGUF_PATH").unwrap_or_else(|_| {
        "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf"
            .to_string()
    })
}

/// `cpu.rs` keeps its `Q4_K`/`Q8_K` layout constants private, so they are
/// re-derived here from `proxima_gguf::quant::q4_k` -- the same source
/// `cpu.rs` derives them from -- rather than pasted as magic numbers.
#[cfg(target_arch = "aarch64")]
const Q4K_BLOCK_BYTES: usize = proxima_gguf::quant::q4_k::BLOCK_BYTES;
#[cfg(target_arch = "aarch64")]
const Q4K_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q4_k::QK_K;
#[cfg(target_arch = "aarch64")]
const Q4K_SUB_BLOCKS: usize = Q4K_BLOCK_ELEMENTS / 32;
#[cfg(target_arch = "aarch64")]
const Q4K_D_OFFSET: usize = 0;
#[cfg(target_arch = "aarch64")]
const Q4K_DMIN_OFFSET: usize = 2;
#[cfg(target_arch = "aarch64")]
const Q4K_SCALES_OFFSET: usize = 4;
#[cfg(target_arch = "aarch64")]
const Q4K_SCALE_BYTES: usize = 12;
#[cfg(target_arch = "aarch64")]
const Q4K_QS_OFFSET: usize = Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES;
#[cfg(target_arch = "aarch64")]
const Q8K_BLOCK_BYTES: usize = 4 + Q4K_BLOCK_ELEMENTS + (Q4K_BLOCK_ELEMENTS / 16) * 2;
#[cfg(target_arch = "aarch64")]
const Q8K_D_OFFSET: usize = 0;
#[cfg(target_arch = "aarch64")]
const Q8K_QS_OFFSET: usize = 4;
#[cfg(target_arch = "aarch64")]
const Q8K_BSUMS_OFFSET: usize = Q8K_QS_OFFSET + Q4K_BLOCK_ELEMENTS;
#[cfg(target_arch = "aarch64")]
const Q8K_BSUMS_COUNT: usize = Q4K_BLOCK_ELEMENTS / 16;

/// openchat-3.5-1210 (Mistral-7B shape) carries 32 decoder blocks; the
/// cold arm round-robins `blk.0..31.attn_q.weight` so the working set
/// (~300 MB) clears every level of this host's cache hierarchy.
#[cfg(target_arch = "aarch64")]
const BLOCK_COUNT: usize = 32;
#[cfg(target_arch = "aarch64")]
const SAMPLES: usize = 9;

#[cfg(target_arch = "aarch64")]
fn f16_le_at(bytes: &[u8], offset: usize) -> f32 {
    let mut raw = [0u8; 2];
    raw.copy_from_slice(&bytes[offset..offset + 2]);
    half::f16::from_le_bytes(raw).to_f32()
}

// ---------------------------------------------------------------- arms

/// Scale/min unpack, verbatim from `cpu.rs`'s `mins_correction_neon`
/// scalar half, split out so `q4k_arm_no_mins` can keep the scales while
/// dropping the bsums reduction.
#[cfg(target_arch = "aarch64")]
fn unpack_scales(scales: &[u8; Q4K_SCALE_BYTES]) -> ([u8; Q4K_SUB_BLOCKS], u32, u32) {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;
    let word_0 = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
    let word_1 = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
    let word_2 = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);
    let mins_lo = word_1 & KMASK1;
    let mins_hi = ((word_2 >> 4) & KMASK2) | (((word_1 >> 6) & KMASK3) << 4);
    let scale_hi = (word_2 & KMASK2) | (((word_0 >> 6) & KMASK3) << 4);
    let scale_lo = word_0 & KMASK1;
    let mut scales_unpacked = [0u8; Q4K_SUB_BLOCKS];
    scales_unpacked[..4].copy_from_slice(&scale_lo.to_le_bytes());
    scales_unpacked[4..].copy_from_slice(&scale_hi.to_le_bytes());
    (scales_unpacked, mins_lo, mins_hi)
}

/// # Safety
/// Caller guarantees FEAT_DotProd and `bsums.len() == Q8K_BSUMS_COUNT * 2`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn mins_reduce(mins_lo: u32, mins_hi: u32, bsums: &[u8]) -> i32 {
    unsafe {
        let mins_words = [mins_lo, mins_hi];
        let mins8 = vld1_u32(mins_words.as_ptr());
        let mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        let bsums_ptr = bsums.as_ptr().cast::<i16>();
        let q8sums = vpaddq_s16(vld1q_s16(bsums_ptr), vld1q_s16(bsums_ptr.add(8)));
        let mins_product = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
        );
        vaddvq_s32(mins_product)
    }
}

/// # Safety
/// Caller guarantees FEAT_DotProd.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    unsafe {
        let result: int32x4_t;
        core::arch::asm!(
            "sdot {result:v}.4s, {a:v}.16b, {b:v}.16b",
            result = inlateout(vreg) acc => result,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
        result
    }
}

/// FULL: byte-for-byte `cpu.rs:5149` `dot_q4k_q8k_block_neon_dotprod`.
///
/// # Safety
/// Caller guarantees FEAT_DotProd and well-formed block slices.
#[cfg(target_arch = "aarch64")]
unsafe fn q4k_arm_full(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q4K_D_OFFSET);
    let dmin_weight = f16_le_at(weight_block, Q4K_DMIN_OFFSET);
    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_block[Q4K_SCALES_OFFSET..Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES]);
    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let bsums = &q8k_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

    let (scales_unpacked, mins_lo, mins_hi) = unpack_scales(&scales);
    let mins_correction = unsafe { mins_reduce(mins_lo, mins_hi, bsums) };

    unsafe {
        let m4b = vdupq_n_u8(0x0f);
        let mzero = vdupq_n_s32(0);
        let q4_base = weight_block[Q4K_QS_OFFSET..].as_ptr();
        let q8_base = q8k_block[Q8K_QS_OFFSET..].as_ptr().cast::<i8>();
        let mut sumi1: i32 = 0;
        let mut sumi2: i32 = 0;
        for j in 0..Q4K_SUB_BLOCKS / 2 {
            let q4bits0 = vld1q_u8(q4_base.add(j * 32));
            let q4bits1 = vld1q_u8(q4_base.add(j * 32 + 16));
            let lo0 = vreinterpretq_s8_u8(vandq_u8(q4bits0, m4b));
            let lo1 = vreinterpretq_s8_u8(vandq_u8(q4bits1, m4b));
            let q8b0 = vld1q_s8(q8_base.add(j * 64));
            let q8b1 = vld1q_s8(q8_base.add(j * 64 + 16));
            let partial_lo = sdot(sdot(mzero, lo0, q8b0), lo1, q8b1);
            sumi1 += vaddvq_s32(partial_lo) * i32::from(scales_unpacked[2 * j]);

            let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits0, 4));
            let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits1, 4));
            let q8b2 = vld1q_s8(q8_base.add(j * 64 + 32));
            let q8b3 = vld1q_s8(q8_base.add(j * 64 + 48));
            let partial_hi = sdot(sdot(mzero, hi0, q8b2), hi1, q8b3);
            sumi2 += vaddvq_s32(partial_hi) * i32::from(scales_unpacked[2 * j + 1]);
        }
        let d = activation_scale * d_weight;
        let dmin = activation_scale * dmin_weight;
        d.mul_add((sumi1 + sumi2) as f32, -(dmin * mins_correction as f32))
    }
}

/// FULL minus the mins correction: no bsums load, no `vpaddq_s16`/
/// `vmull_s16` reduce, no `dmin` f16 decode, no final subtract. Scales are
/// still unpacked (the dot needs them), so the delta is exactly the mins
/// phase.
///
/// # Safety
/// Caller guarantees FEAT_DotProd and well-formed block slices.
#[cfg(target_arch = "aarch64")]
unsafe fn q4k_arm_no_mins(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q4K_D_OFFSET);
    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_block[Q4K_SCALES_OFFSET..Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES]);
    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let (scales_unpacked, _, _) = unpack_scales(&scales);

    unsafe {
        let m4b = vdupq_n_u8(0x0f);
        let mzero = vdupq_n_s32(0);
        let q4_base = weight_block[Q4K_QS_OFFSET..].as_ptr();
        let q8_base = q8k_block[Q8K_QS_OFFSET..].as_ptr().cast::<i8>();
        let mut sumi1: i32 = 0;
        let mut sumi2: i32 = 0;
        for j in 0..Q4K_SUB_BLOCKS / 2 {
            let q4bits0 = vld1q_u8(q4_base.add(j * 32));
            let q4bits1 = vld1q_u8(q4_base.add(j * 32 + 16));
            let lo0 = vreinterpretq_s8_u8(vandq_u8(q4bits0, m4b));
            let lo1 = vreinterpretq_s8_u8(vandq_u8(q4bits1, m4b));
            let q8b0 = vld1q_s8(q8_base.add(j * 64));
            let q8b1 = vld1q_s8(q8_base.add(j * 64 + 16));
            let partial_lo = sdot(sdot(mzero, lo0, q8b0), lo1, q8b1);
            sumi1 += vaddvq_s32(partial_lo) * i32::from(scales_unpacked[2 * j]);

            let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits0, 4));
            let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits1, 4));
            let q8b2 = vld1q_s8(q8_base.add(j * 64 + 32));
            let q8b3 = vld1q_s8(q8_base.add(j * 64 + 48));
            let partial_hi = sdot(sdot(mzero, hi0, q8b2), hi1, q8b3);
            sumi2 += vaddvq_s32(partial_hi) * i32::from(scales_unpacked[2 * j + 1]);
        }
        (activation_scale * d_weight) * (sumi1 + sumi2) as f32
    }
}

/// FULL minus the per-sub-block scale application: the eight
/// `vaddvq_s32` horizontal reduces and eight scalar multiplies collapse to
/// one vector accumulator plus a single reduce at the end. Scales are
/// still unpacked and the mins correction still runs, so the delta is
/// exactly the scale-application phase.
///
/// # Safety
/// Caller guarantees FEAT_DotProd and well-formed block slices.
#[cfg(target_arch = "aarch64")]
unsafe fn q4k_arm_no_scale_apply(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q4K_D_OFFSET);
    let dmin_weight = f16_le_at(weight_block, Q4K_DMIN_OFFSET);
    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_block[Q4K_SCALES_OFFSET..Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES]);
    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let bsums = &q8k_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];
    let (scales_unpacked, mins_lo, mins_hi) = unpack_scales(&scales);
    let mins_correction = unsafe { mins_reduce(mins_lo, mins_hi, bsums) };
    black_box(scales_unpacked);

    unsafe {
        let m4b = vdupq_n_u8(0x0f);
        let mut acc = vdupq_n_s32(0);
        let q4_base = weight_block[Q4K_QS_OFFSET..].as_ptr();
        let q8_base = q8k_block[Q8K_QS_OFFSET..].as_ptr().cast::<i8>();
        for j in 0..Q4K_SUB_BLOCKS / 2 {
            let q4bits0 = vld1q_u8(q4_base.add(j * 32));
            let q4bits1 = vld1q_u8(q4_base.add(j * 32 + 16));
            let lo0 = vreinterpretq_s8_u8(vandq_u8(q4bits0, m4b));
            let lo1 = vreinterpretq_s8_u8(vandq_u8(q4bits1, m4b));
            let q8b0 = vld1q_s8(q8_base.add(j * 64));
            let q8b1 = vld1q_s8(q8_base.add(j * 64 + 16));
            acc = sdot(sdot(acc, lo0, q8b0), lo1, q8b1);

            let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits0, 4));
            let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits1, 4));
            let q8b2 = vld1q_s8(q8_base.add(j * 64 + 32));
            let q8b3 = vld1q_s8(q8_base.add(j * 64 + 48));
            acc = sdot(sdot(acc, hi0, q8b2), hi1, q8b3);
        }
        let d = activation_scale * d_weight;
        let dmin = activation_scale * dmin_weight;
        d.mul_add(vaddvq_s32(acc) as f32, -(dmin * mins_correction as f32))
    }
}

/// FULL minus the nibble unpack ALU: the eight `vandq_u8` and eight
/// `vshrq_n_u8` are dropped and the raw packed byte vector is fed to
/// `sdot` twice. IDENTICAL loads, identical memory traffic, intentionally
/// wrong values -- this arm isolates the mask/shift ALU cost alone, with
/// no bandwidth confound.
///
/// # Safety
/// Caller guarantees FEAT_DotProd and well-formed block slices.
#[cfg(target_arch = "aarch64")]
unsafe fn q4k_arm_no_nibble_alu(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q4K_D_OFFSET);
    let dmin_weight = f16_le_at(weight_block, Q4K_DMIN_OFFSET);
    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_block[Q4K_SCALES_OFFSET..Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES]);
    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let bsums = &q8k_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];
    let (scales_unpacked, mins_lo, mins_hi) = unpack_scales(&scales);
    let mins_correction = unsafe { mins_reduce(mins_lo, mins_hi, bsums) };

    unsafe {
        let mzero = vdupq_n_s32(0);
        let q4_base = weight_block[Q4K_QS_OFFSET..].as_ptr();
        let q8_base = q8k_block[Q8K_QS_OFFSET..].as_ptr().cast::<i8>();
        let mut sumi1: i32 = 0;
        let mut sumi2: i32 = 0;
        for j in 0..Q4K_SUB_BLOCKS / 2 {
            let q4bits0 = vld1q_u8(q4_base.add(j * 32));
            let q4bits1 = vld1q_u8(q4_base.add(j * 32 + 16));
            let lo0 = vreinterpretq_s8_u8(q4bits0);
            let lo1 = vreinterpretq_s8_u8(q4bits1);
            let q8b0 = vld1q_s8(q8_base.add(j * 64));
            let q8b1 = vld1q_s8(q8_base.add(j * 64 + 16));
            let partial_lo = sdot(sdot(mzero, lo0, q8b0), lo1, q8b1);
            sumi1 += vaddvq_s32(partial_lo) * i32::from(scales_unpacked[2 * j]);

            let q8b2 = vld1q_s8(q8_base.add(j * 64 + 32));
            let q8b3 = vld1q_s8(q8_base.add(j * 64 + 48));
            let partial_hi = sdot(sdot(mzero, lo0, q8b2), lo1, q8b3);
            sumi2 += vaddvq_s32(partial_hi) * i32::from(scales_unpacked[2 * j + 1]);
        }
        let d = activation_scale * d_weight;
        let dmin = activation_scale * dmin_weight;
        d.mul_add((sumi1 + sumi2) as f32, -(dmin * mins_correction as f32))
    }
}

/// SDOT ONLY: the 8 weight loads, 16 activation loads and 16 `sdot`s into
/// one vector accumulator. No f16 decode, no scale unpack, no mins, no
/// per-sub-block reduce.
///
/// # Safety
/// Caller guarantees FEAT_DotProd and well-formed block slices.
#[cfg(target_arch = "aarch64")]
unsafe fn q4k_arm_sdot_only(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    unsafe {
        let mut acc = vdupq_n_s32(0);
        let q4_base = weight_block[Q4K_QS_OFFSET..].as_ptr();
        let q8_base = q8k_block[Q8K_QS_OFFSET..].as_ptr().cast::<i8>();
        for j in 0..Q4K_SUB_BLOCKS / 2 {
            let q4bits0 = vreinterpretq_s8_u8(vld1q_u8(q4_base.add(j * 32)));
            let q4bits1 = vreinterpretq_s8_u8(vld1q_u8(q4_base.add(j * 32 + 16)));
            let q8b0 = vld1q_s8(q8_base.add(j * 64));
            let q8b1 = vld1q_s8(q8_base.add(j * 64 + 16));
            let q8b2 = vld1q_s8(q8_base.add(j * 64 + 32));
            let q8b3 = vld1q_s8(q8_base.add(j * 64 + 48));
            acc = sdot(sdot(acc, q4bits0, q8b0), q4bits1, q8b1);
            acc = sdot(sdot(acc, q4bits0, q8b2), q4bits1, q8b3);
        }
        vaddvq_s32(acc) as f32
    }
}

/// TOUCH ONLY control: issues the same 8 weight loads and 16 activation
/// loads and consumes them with 23 `vaddq_u8`s so nothing is eliminated.
/// This is the traffic floor, NOT zero -- the 23 adds are the minimum
/// price of consuming the loads, so the floor it reports is an upper bound
/// on pure memory cost.
///
/// # Safety
/// Caller guarantees well-formed block slices.
#[cfg(target_arch = "aarch64")]
unsafe fn q4k_arm_touch_only(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    unsafe {
        let mut acc: uint8x16_t = vdupq_n_u8(0);
        let q4_base = weight_block[Q4K_QS_OFFSET..].as_ptr();
        let q8_base = q8k_block[Q8K_QS_OFFSET..].as_ptr();
        for j in 0..Q4K_SUB_BLOCKS / 2 {
            acc = vaddq_u8(acc, vld1q_u8(q4_base.add(j * 32)));
            acc = vaddq_u8(acc, vld1q_u8(q4_base.add(j * 32 + 16)));
            acc = vaddq_u8(acc, vld1q_u8(q8_base.add(j * 64)));
            acc = vaddq_u8(acc, vld1q_u8(q8_base.add(j * 64 + 16)));
            acc = vaddq_u8(acc, vld1q_u8(q8_base.add(j * 64 + 32)));
            acc = vaddq_u8(acc, vld1q_u8(q8_base.add(j * 64 + 48)));
        }
        f32::from(acc_first_byte(acc))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn acc_first_byte(acc: uint8x16_t) -> u8 {
    let mut out = [0u8; 16];
    // SAFETY: `out` is exactly one 128-bit vector wide.
    unsafe { core::arch::aarch64::vst1q_u8(out.as_mut_ptr(), acc) };
    out[0]
}

// ------------------------------------------- disassembly-only copies
//
// The timed arms above carry NO attributes, exactly like `cpu.rs`'s own
// `dot_q4k_q8k_block_neon_dotprod`, so LLVM inlines them into the block
// loop the same way it inlines the shipped kernel into `dot_q4k_q8k`. An
// earlier revision marked them `#[inline(never)]` to make instruction
// counting easy and paid +34.5% for it -- the call/return plus the slice
// bounds checks that inlining hoists out of the loop -- which broke the
// "must track the shipped kernel" gate outright. So the countable copies
// live here instead, called once outside every timed region purely to
// keep them in the binary.

macro_rules! countable {
    ($wrapper:ident, $arm:path) => {
        /// # Safety
        /// Caller guarantees FEAT_DotProd and well-formed block slices.
        #[cfg(target_arch = "aarch64")]
        #[unsafe(no_mangle)]
        #[inline(never)]
        pub unsafe fn $wrapper(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
            // SAFETY: forwarded from this function's own precondition.
            unsafe { $arm(weight_block, q8k_block) }
        }
    };
}

countable!(q4k_count_full, q4k_arm_full);
countable!(q4k_count_no_mins, q4k_arm_no_mins);
countable!(q4k_count_no_scale_apply, q4k_arm_no_scale_apply);
countable!(q4k_count_no_nibble_alu, q4k_arm_no_nibble_alu);
countable!(q4k_count_sdot_only, q4k_arm_sdot_only);
countable!(q4k_count_touch_only, q4k_arm_touch_only);

/// Calls every `#[inline(never)]` countable copy once so none is dropped
/// as dead code before `objdump` can read its instruction count.
///
/// # Safety
/// Caller guarantees FEAT_DotProd and well-formed block slices.
#[cfg(target_arch = "aarch64")]
unsafe fn keep_countable_copies(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    // SAFETY: forwarded from this function's own precondition.
    unsafe {
        q4k_count_full(weight_block, q8k_block)
            + q4k_count_no_mins(weight_block, q8k_block)
            + q4k_count_no_scale_apply(weight_block, q8k_block)
            + q4k_count_no_nibble_alu(weight_block, q8k_block)
            + q4k_count_sdot_only(weight_block, q8k_block)
            + q4k_count_touch_only(weight_block, q8k_block)
    }
}

// ------------------------------------------------------------- harness

#[cfg(target_arch = "aarch64")]
struct Stat {
    mean: f64,
    cov: f64,
    min: f64,
    max: f64,
}

#[cfg(target_arch = "aarch64")]
impl Stat {
    fn from(samples: &[f64]) -> Self {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / samples.len() as f64;
        Self {
            mean,
            cov: variance.sqrt() / mean,
            min: samples.iter().copied().fold(f64::INFINITY, f64::min),
            max: samples.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn time_pass<F: FnMut() -> f32>(mut body: F, macs_per_pass: u64) -> Stat {
    black_box(body());
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        let out = body();
        let elapsed = start.elapsed();
        black_box(out);
        samples.push(elapsed.as_secs_f64() * 1e9 / macs_per_pass as f64);
    }
    Stat::from(&samples)
}

#[cfg(target_arch = "aarch64")]
fn report(label: &str, stat: &Stat, baseline: Option<&Stat>) {
    let delta = baseline.map_or(String::new(), |full| {
        let delta_ns = full.mean - stat.mean;
        format!(
            "  delta_vs_full={delta_ns:+.5} ns/mac ({:+.1}% of full)",
            100.0 * delta_ns / full.mean
        )
    });
    println!(
        "{label:<34} {:.5} ns/mac  cov={:.2}%  [{:.5}, {:.5}]{delta}",
        stat.mean,
        stat.cov * 100.0,
        stat.min,
        stat.max
    );
}

#[cfg(target_arch = "aarch64")]
fn parse_header(path: &Path) -> (ParsedGguf, u64) {
    let mut file = File::open(path).expect("open real gguf file");
    let file_len = file.metadata().expect("stat gguf file").len();
    let mut prefix_len = 1usize << 20;
    loop {
        let mut buf = vec![0u8; prefix_len];
        file.seek(SeekFrom::Start(0)).expect("seek to start");
        let read = file.read(&mut buf).expect("read gguf prefix");
        buf.truncate(read);
        if let Ok((parser, events)) = GgufParser::new().push(&buf) {
            let mut version = None;
            let mut metadata = Vec::new();
            let mut tensors = Vec::new();
            let mut completion = None;
            for event in events {
                match event {
                    GgufEvent::Header {
                        version: version_value,
                        ..
                    } => version = Some(version_value),
                    GgufEvent::Metadata { key, value } => metadata.push((key, value)),
                    GgufEvent::Tensor(tensor) => tensors.push(tensor),
                    GgufEvent::Complete {
                        data_offset,
                        alignment,
                    } => {
                        completion = Some((data_offset, alignment));
                    }
                }
            }
            if let (Some(version), Some((data_offset, alignment))) = (version, completion) {
                parser.finish().expect("parser reports complete and clean");
                return (
                    ParsedGguf {
                        version,
                        tensor_count: tensors.len() as u64,
                        kv_count: metadata.len() as u64,
                        metadata,
                        tensors,
                        data_offset,
                        alignment,
                    },
                    file_len,
                );
            }
        }
        assert!(
            prefix_len < (1 << 26),
            "gguf header/directory exceeded 64 MiB prefix budget"
        );
        prefix_len *= 2;
    }
}

#[cfg(target_arch = "aarch64")]
fn find_tensor<'a>(parsed: &'a ParsedGguf, name: &str) -> &'a TensorInfo {
    parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("tensor {name} not found in real gguf file"))
}

#[cfg(target_arch = "aarch64")]
fn read_tensor_bytes(
    file: &mut File,
    parsed: &ParsedGguf,
    tensor: &TensorInfo,
    file_len: u64,
) -> Vec<u8> {
    let range = parsed
        .tensor_data_range(tensor, file_len)
        .expect("tensor byte range within file bounds");
    let mut buf = vec![0u8; (range.end - range.start) as usize];
    file.seek(SeekFrom::Start(range.start))
        .expect("seek to tensor data");
    file.read_exact(&mut buf)
        .expect("read exact tensor byte range");
    buf
}

#[cfg(target_arch = "aarch64")]
macro_rules! arm_pass {
    ($buffers:expr, $activation:expr, $row_bytes:expr, $repeats:expr, $arm:path) => {
        || {
            let mut total = 0.0f32;
            for _ in 0..$repeats {
                for buffer in $buffers.iter() {
                    for row in buffer.chunks_exact($row_bytes) {
                        let mut acc = 0.0f32;
                        for (weight_block, q8k_block) in row
                            .chunks_exact(Q4K_BLOCK_BYTES)
                            .zip($activation.chunks_exact(Q8K_BLOCK_BYTES))
                        {
                            // SAFETY: aarch64 build implies FEAT_DotProd on every
                            // target this workspace builds for (build.rs
                            // `emit_dotprod_cfg`); `chunks_exact` guarantees both
                            // slices are exactly one super-block wide, which is
                            // `dot_q4k_q8k`'s own argument for the same call.
                            acc += unsafe { $arm(weight_block, q8k_block) };
                        }
                        total += acc;
                    }
                }
            }
            total
        }
    };
}

#[cfg(target_arch = "aarch64")]
fn run_configuration(
    label: &str,
    buffers: &[Vec<u8>],
    activation_q8k: &[u8],
    blocks_per_row: usize,
    repeats: u64,
) {
    let macs: u64 = buffers
        .iter()
        .map(|buffer| (buffer.len() / Q4K_BLOCK_BYTES) as u64)
        .sum::<u64>()
        * Q4K_BLOCK_ELEMENTS as u64
        * repeats;
    let bytes: usize = buffers.iter().map(Vec::len).sum();
    println!(
        "\n=== {label}: {} buffers, {bytes} weight bytes ({:.1} MiB), {macs} macs/pass, single thread ===",
        buffers.len(),
        bytes as f64 / (1024.0 * 1024.0)
    );

    let row_bytes = blocks_per_row * Q4K_BLOCK_BYTES;
    let real = time_pass(
        || {
            let mut total = 0.0f32;
            for _ in 0..repeats {
                for buffer in buffers {
                    for row in buffer.chunks_exact(row_bytes) {
                        total += dot_q4k_q8k(row, activation_q8k).unwrap();
                    }
                }
            }
            total
        },
        macs,
    );

    let full = time_pass(
        arm_pass!(buffers, activation_q8k, row_bytes, repeats, q4k_arm_full),
        macs,
    );
    report("real_dot_q4k_q8k (shipped)", &real, None);
    report("local_full (reconstruction)", &full, None);
    println!(
        "reconstruction bias vs shipped: {:+.2}% (gate: local_full must track the shipped kernel)",
        100.0 * (full.mean - real.mean) / real.mean
    );

    let arms: [(&str, Stat); 5] = [
        (
            "minus_mins_correction",
            time_pass(
                arm_pass!(buffers, activation_q8k, row_bytes, repeats, q4k_arm_no_mins),
                macs,
            ),
        ),
        (
            "minus_scale_application",
            time_pass(
                arm_pass!(
                    buffers,
                    activation_q8k,
                    row_bytes,
                    repeats,
                    q4k_arm_no_scale_apply
                ),
                macs,
            ),
        ),
        (
            "minus_nibble_mask_shift",
            time_pass(
                arm_pass!(
                    buffers,
                    activation_q8k,
                    row_bytes,
                    repeats,
                    q4k_arm_no_nibble_alu
                ),
                macs,
            ),
        ),
        (
            "sdot_loop_only",
            time_pass(
                arm_pass!(
                    buffers,
                    activation_q8k,
                    row_bytes,
                    repeats,
                    q4k_arm_sdot_only
                ),
                macs,
            ),
        ),
        (
            "touch_only_control",
            time_pass(
                arm_pass!(
                    buffers,
                    activation_q8k,
                    row_bytes,
                    repeats,
                    q4k_arm_touch_only
                ),
                macs,
            ),
        ),
    ];
    for (name, stat) in &arms {
        report(name, stat, Some(&full));
    }

    let phase_sum: f64 = arms[..3]
        .iter()
        .map(|(_, stat)| full.mean - stat.mean)
        .sum();
    let floor = arms[4].1.mean;
    println!(
        "accounting: touch_floor {floor:.5} + named_phases {phase_sum:.5} = {:.5}; full {:.5}; RESIDUAL {:+.5} ns/mac ({:+.1}% of full)",
        floor + phase_sum,
        full.mean,
        full.mean - floor - phase_sum,
        100.0 * (full.mean - floor - phase_sum) / full.mean
    );
}

#[cfg(target_arch = "aarch64")]
fn bench_quantize(in_dims: &[usize]) {
    println!(
        "\n=== quantize_row_q8k: per-call cost of the once-per-matmul activation quantize ==="
    );
    for &in_dim in in_dims {
        let mut lcg = Lcg(99);
        let activation: Vec<f32> = (0..in_dim).map(|_| lcg.next_unit() * 0.5).collect();
        let mut out = vec![0u8; (in_dim / Q4K_BLOCK_ELEMENTS) * Q8K_BLOCK_BYTES];
        const CALLS: u64 = 2000;
        let stat = time_pass(
            || {
                for _ in 0..CALLS {
                    quantize_row_q8k(&activation, &mut out).unwrap();
                }
                out[0] as f32
            },
            CALLS,
        );
        println!(
            "quantize_row_q8k in_dim={in_dim:<6} {:.1} ns/call  cov={:.2}%  [{:.1}, {:.1}]",
            stat.mean,
            stat.cov * 100.0,
            stat.min,
            stat.max
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn main() {
    let path = gguf_path();
    if !Path::new(&path).exists() {
        println!("real gguf file not found at {path}; nothing to bench");
        return;
    }
    let (parsed, file_len) = parse_header(Path::new(&path));
    let mut file = File::open(&path).expect("reopen real gguf file for tensor data");

    let warm_tensor = find_tensor(&parsed, "blk.0.attn_q.weight");
    let in_dim = warm_tensor.dims[0] as usize;
    let out_dim = warm_tensor.dims[1] as usize;
    let blocks_per_row = in_dim / Q4K_BLOCK_ELEMENTS;
    println!(
        "tensor blk.N.attn_q.weight dims=[{in_dim}, {out_dim}] blocks_per_row={blocks_per_row}"
    );

    let mut cold_buffers: Vec<Vec<u8>> = Vec::with_capacity(BLOCK_COUNT);
    for block in 0..BLOCK_COUNT {
        let name = format!("blk.{block}.attn_q.weight");
        let tensor = find_tensor(&parsed, &name);
        assert_eq!(
            tensor.dims[0] as usize, in_dim,
            "{name}: in_dim shape drift vs blk.0"
        );
        cold_buffers.push(read_tensor_bytes(&mut file, &parsed, tensor, file_len));
    }
    let warm_buffers = vec![cold_buffers[0].clone()];

    let mut lcg = Lcg(4242);
    let activation: Vec<f32> = (0..in_dim).map(|_| lcg.next_unit() * 0.5).collect();
    let mut activation_q8k = vec![0u8; blocks_per_row * Q8K_BLOCK_BYTES];
    quantize_row_q8k(&activation, &mut activation_q8k).expect("activation quantizes");

    // correctness gate: the reconstruction must reproduce the shipped
    // kernel bit-for-bit on real bytes before any timing is admissible.
    let row_bytes = blocks_per_row * Q4K_BLOCK_BYTES;
    let mut checked_rows = 0usize;
    for row in cold_buffers[0].chunks_exact(row_bytes).take(64) {
        let shipped = dot_q4k_q8k(row, &activation_q8k).unwrap();
        let mut local = 0.0f32;
        for (index, weight_block) in row.as_chunks::<Q4K_BLOCK_BYTES>().0.iter().enumerate() {
            let q8k_block = &activation_q8k[index * Q8K_BLOCK_BYTES..(index + 1) * Q8K_BLOCK_BYTES];
            // SAFETY: aarch64 implies FEAT_DotProd here; slices are one super-block each.
            local += unsafe { q4k_arm_full(weight_block, q8k_block) };
        }
        assert!(
            (shipped - local).abs() <= shipped.abs() * 1e-6 + 1e-4,
            "reconstruction diverged from shipped kernel: {shipped} vs {local}"
        );
        checked_rows += 1;
    }
    println!(
        "reconstruction agrees with dot_q4k_q8k on {checked_rows} real rows (N asserted, not assumed)"
    );
    // SAFETY: aarch64 implies FEAT_DotProd here; both slices are one super-block wide.
    let countable = unsafe {
        keep_countable_copies(
            &cold_buffers[0][..Q4K_BLOCK_BYTES],
            &activation_q8k[..Q8K_BLOCK_BYTES],
        )
    };
    println!("countable copies retained for objdump (sum {countable:.3}, never timed)");

    // WARM repeats 32x so both configurations time the same total work
    // (~537M macs/sample); a single 16.8M-mac pass is ~0.4 ms and far too
    // short to be stable against ambient scheduling.
    run_configuration(
        "WARM (single 9.0 MiB tensor, 32 repeats)",
        &warm_buffers,
        &activation_q8k,
        blocks_per_row,
        BLOCK_COUNT as u64,
    );
    run_configuration(
        "COLD (32 distinct tensors)",
        &cold_buffers,
        &activation_q8k,
        blocks_per_row,
        1,
    );
    bench_quantize(&[4096, 14336]);
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    println!("q4k super-block phase ablation is aarch64/FEAT_DotProd only; nothing to bench here");
}
