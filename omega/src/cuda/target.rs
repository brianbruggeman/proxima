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
//!   mask; HIP's take a 64-bit one on every wavefront width, and the header
//!   static-asserts that size even when the wave is 32 lanes. The value is the
//!   wave's lanes, from [`proxima_numeric::all_lanes_mask`]; only its spelling
//!   differs here.
//!
//! A fourth difference is a capability, not a dialect fact: inline PTX
//! (`mma.sync`) has no HIP equivalent in this crate, so [`Target::card_asm`]
//! is false for HIP and every body that emits inline assembly is not selected
//! for it. Those bodies are named where the selection happens
//! ([`crate::cuda`]'s `text_plan_token`, `reject_unserved_prequantized`,
//! `prologue_kernels`) and HIP renders the generic per-element / cooperative
//! path instead — the same path CUDA's own v1 scope documents.

use alloc::string::String;

use proxima_numeric::all_lanes_mask;
use proxima_numeric::codec::Codec;

use crate::identity::KernelLanguage;

/// The AMD wavefront width every CDNA part (gfx90a, gfx942, the MI series)
/// runs at. RDNA parts run 32-lane waves and are emitted at
/// [`Target::hip_wave`]`(32)`; both are hardware facts, never a policy knob,
/// so neither is read from the device at emit time — emission has no device
/// handle, and the driver must launch at the width the text was built for.
pub const HIP_WAVE64: u64 = 64;

/// The name every emitted `_sync` shuffle passes as its mask. The preamble
/// defines it for the active target, so the emitted text carries one symbolic
/// all-lanes value rather than a literal each renderer could get wrong.
pub(crate) const LANE_MASK_TOKEN: &str = "PROXIMA_ALL_LANES";

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
    /// the numeric format name the tensor-core instruction takes as a multiplicand
    /// (`[contraction] cuda_mma_operand`); empty where `card_asm` is false.
    pub mma_operand: &'static str,
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
        mma_operand: crate::sized::CUDA_MMA_OPERAND,
    };

    /// AMD HIP C++ at the CDNA wavefront width.
    pub const HIP: Self = Self {
        language: KernelLanguage::Hip,
        wave: HIP_WAVE64,
        fp16_header: "hip/hip_fp16.h",
        bf16_header: "hip/hip_bf16.h",
        bfloat_type: "__hip_bfloat16",
        card_asm: false,
        mma_operand: "",
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

    /// The dialect's name as an error names the backend that refused.
    pub const fn language_name(&self) -> &'static str {
        match self.language {
            KernelLanguage::Cuda => "cuda",
            KernelLanguage::Hip => "hip",
            KernelLanguage::Metal => "metal",
            KernelLanguage::Wgsl => "wgsl",
        }
    }

    /// The codec of [`Self::mma_operand`], the format a plain float operand is split into for the tensor-core tile.
    pub fn mma_operand_codec(&self) -> Option<Codec> {
        Codec::from_name(self.mma_operand)
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

    /// The all-lanes-active mask the `_sync` shuffle intrinsics require. The
    /// preamble defines this token from [`Self::lane_mask_literal`], so no
    /// renderer spells a mask of its own and the dialect's type cannot drift
    /// away from a call site.
    pub const fn lane_mask(&self) -> &'static str {
        LANE_MASK_TOKEN
    }

    /// The typed all-lanes literal the preamble binds `LANE_MASK_TOKEN` to.
    /// HIP's `_sync` mask is `unsigned long long` on every wavefront width, so
    /// a 32-lane wave still emits a 64-bit-typed value with its 32 lanes set.
    pub fn lane_mask_literal(&self) -> String {
        let lanes = u32::try_from(self.wave).unwrap_or(u32::MAX);
        let mask = all_lanes_mask(lanes);
        if self.language == KernelLanguage::Hip {
            alloc::format!("{mask:#x}ull")
        } else {
            alloc::format!("{mask:#x}u")
        }
    }

    /// The one `#define` that binds [`LANE_MASK_TOKEN`] for this target. A
    /// renderer that builds its source without [`crate::cuda::preamble`] must
    /// still start from this, or its `_sync` call names an undefined token.
    pub(crate) fn lane_mask_define(&self) -> String {
        alloc::format!("{}#define {LANE_MASK_TOKEN} {}\n", self.wave_width_guard(), self.lane_mask_literal())
    }

    /// For HIP, the preprocessor guard that refuses text built for another wavefront width: hipRTC
    /// compiles an RDNA part at 32 lanes, and 64-lane text on it drops half of every shuffle tree.
    /// Empty for CUDA, whose warp is always 32.
    pub(crate) fn wave_width_guard(&self) -> String {
        if self.language != KernelLanguage::Hip {
            return String::new();
        }
        alloc::format!(
                "#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != {wave}\n#error \"omega HIP text built for {wave}-lane waves\"\n#endif\n",
                wave = self.wave
            )
    }

    /// For HIP, the kernel-entry check that traps when the device runs another width than the text
    /// was built for; `__AMDGCN_WAVEFRONT_SIZE` is deprecated in ROCm 7, so the runtime `warpSize`
    /// backs the preprocessor guard. Empty for CUDA.
    pub(crate) fn entry_wave_guard(&self) -> String {
        if self.language != KernelLanguage::Hip {
            return String::new();
        }
        alloc::format!("    if (warpSize != {}) {{ __builtin_trap(); }}\n", self.wave)
    }

    /// A complete statement that broadcasts `value` from `lane` into `dest`.
    /// The shuffle is only ever rendered with its destination, so a caller
    /// cannot emit the call as a discarded bare expression.
    pub(crate) fn shfl_broadcast_into(&self, dest: &str, value: &str, lane: &str) -> String {
        alloc::format!("{dest} = __shfl_sync({}, {value}, {lane});", self.lane_mask())
    }

    /// The `warp_partials[N]` extent a block-wide merge needs: one slot per
    /// warp in a full block.
    pub const fn max_warps_per_block(&self) -> u64 {
        crate::cuda::MAX_BLOCK_WIDTH / self.wave
    }
}
