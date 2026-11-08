//! What differs between the two card-C dialects this crate emits: NVIDIA's
//! CUDA C and AMD's HIP C++.
//!
//! HIP is a source-level reimplementation of CUDA's kernel language, not a
//! similar-but-different language: `__global__`, `__shared__`, `half2`,
//! `__shfl_down_sync`, `atomicOr` and `__ldg` all keep their spelling. Three
//! things do NOT, and each is a value here rather than a second renderer:
//!
//! - **the wavefront width.** A warp is 32 lanes on every NVIDIA part and 64
//!   on a CDNA wavefront (32 on RDNA). Every lane index, shuffle tree depth,
//!   warp-crossing merge and cooperative-reduce block width in this module is
//!   derived from [`Target::wave`], so one body renders correctly for both.
//! - **the include and the bfloat spelling.** `<cuda_fp16.h>` vs
//!   `<hip/hip_fp16.h>`; `__nv_bfloat16` vs `__hip_bfloat16`.
//! - **the warp-shuffle mask type.** CUDA's `_sync` intrinsics take a 32-bit
//!   mask; HIP's take a 64-bit one on a wave64 target.
//!
//! A fourth difference is a capability, not a dialect fact: inline PTX
//! (`mma.sync`) has no HIP equivalent in this crate, so [`Target::card_asm`]
//! is false for HIP and every body that emits inline assembly is not selected
//! for it. Those bodies are named where the selection happens
//! ([`crate::cuda`]'s `text_plan_token`, `reject_unserved_prequantized`,
//! `prologue_kernels`) and HIP renders the generic per-element / cooperative
//! path instead — the same path CUDA's own v1 scope documents.

use alloc::string::String;

use crate::identity::KernelLanguage;

/// The AMD wavefront width every CDNA part (gfx90a, gfx942, the MI series)
/// runs at. RDNA parts run 32-lane waves and are emitted at
/// [`Target::hip_wave`]`(32)`; both are hardware facts, never a policy knob,
/// so neither is read from the device at emit time — emission has no device
/// handle, and the driver must launch at the width the text was built for.
pub const HIP_WAVE64: u64 = 64;

/// One emission target: the card-C dialect plus the hardware wavefront width
/// the emitted text's lane arithmetic is built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub language: KernelLanguage,
    /// lanes per warp/wavefront. 32 for every NVIDIA part; see [`HIP_WAVE64`].
    pub wave: u64,
    pub fp16_header: &'static str,
    pub bf16_header: &'static str,
    /// the storage type a bfloat16 operand is read as
    pub bfloat_type: &'static str,
    /// whether this target's toolchain accepts the inline card assembly
    /// (`mma.sync`) and the intrinsic sets those bodies are built on. False
    /// makes the emitter choose the generic body instead of refusing, which is
    /// what lets the same `BoundOp` graph render for both vendors.
    pub card_asm: bool,
}

impl Target {
    /// NVIDIA CUDA C: 32-lane warps, `cuda_fp16.h`, `__nv_bfloat16`, inline
    /// `mma.sync` available.
    pub const CUDA: Self = Self {
        language: KernelLanguage::Cuda,
        wave: 32,
        fp16_header: "cuda_fp16.h",
        bf16_header: "cuda_bf16.h",
        bfloat_type: "__nv_bfloat16",
        card_asm: true,
    };

    /// AMD HIP C++ at the CDNA wavefront width.
    pub const HIP: Self = Self {
        language: KernelLanguage::Hip,
        wave: HIP_WAVE64,
        fp16_header: "hip/hip_fp16.h",
        bf16_header: "hip/hip_bf16.h",
        bfloat_type: "__hip_bfloat16",
        card_asm: false,
    };

    /// HIP at an explicit wavefront width: [`HIP_WAVE64`] for CDNA, 32 for
    /// RDNA. The width is baked into lane indexing, so a driver that launches
    /// text built for the other width produces wrong results rather than a
    /// slow launch — the caller states which part it is targeting.
    pub const fn hip_wave(wave: u64) -> Self {
        let mut target = Self::HIP;
        target.wave = wave;
        target
    }

    /// `warpSize - 1`, the mask that isolates a lane index inside its wave.
    pub const fn lane_bits(&self) -> u64 {
        self.wave - 1
    }

    /// Folded into every entry name so a compiled module for one dialect can
    /// never be served for another with otherwise identical structure. Empty
    /// for the target whose entry names predate this seam, so CUDA's cached
    /// modules and fixtures keep their identities.
    pub const fn entry_token(&self) -> &'static str {
        match (self.language, self.wave) {
            (KernelLanguage::Cuda, _) => "",
            (_, 64) => "_hip",
            (_, _) => "_hip_w32",
        }
    }

    /// `log2(warpSize)`: the shift that extracts the warp index from a thread
    /// id, and the depth of the butterfly shuffle.
    pub const fn warp_shift(&self) -> u64 {
        match self.wave {
            32 => 5,
            64 => 6,
            _ => 0,
        }
    }

    /// The all-lanes-active mask the `_sync` shuffle intrinsics require, in
    /// the width this dialect's intrinsics take.
    pub const fn lane_mask(&self) -> &'static str {
        if self.wave == 64 { "0xffffffffffffffffull" } else { "0xffffffffu" }
    }

    pub(crate) fn shfl_broadcast(&self, value: &str, lane: &str) -> String {
        alloc::format!("__shfl_sync({}, {value}, {lane})", self.lane_mask())
    }

    /// The `warp_partials[N]` extent a block-wide merge needs: one slot per
    /// warp in a full block.
    pub const fn max_warps_per_block(&self) -> u64 {
        crate::cuda::MAX_BLOCK_WIDTH / self.wave
    }
}
