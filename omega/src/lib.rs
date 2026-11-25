//! Plan and execute tensor programs on a CPU and on a GPU.
//!
//! A program is data: an ordered list of [`Op`](proxima_tensor::Op)s over named
//! buffers. [`backend::plan_named_with_config`] turns that program plus a
//! [`config::RuntimeConfig`] into a backend-specific plan, and
//! [`backend::execute_plan_named`] runs it against the bound data.
//!
//! [`backend::Engine`] has two variants, `Cpu` and `Gpu`, because there are two
//! places an op runs; [`backend::GpuDriver`] is the second question, which
//! driver renders the GPU (`Metal`, `Wgpu`, and with `cuda-driver`, `Cuda` and
//! `Hip`). Both are config fields, so moving a program between engines is a
//! config edit rather than a code change.
//!
//! The CPU half is `proxima_tensor::cpu`, the same bound descriptor executed by
//! nested loops. The GPU half emits Metal Shading Language ([`msl::emit`]) or
//! CUDA C / HIP C++ from that same descriptor and drives it through the
//! platform API; [`msl::Kernel`] and [`msl::Binding`] are where an emitter
//! stops, and compiling, allocating and dispatching are the driver's job.
//! Neither backend's shape leaks into `proxima-tensor`: `bind::bind` says only
//! what to compute, never how.
//!
//! # Tiers
//!
//! bare (no_std, no alloc): [`error`] and [`sized`] only -- [`EmitError`]'s
//! variants and the build-time-configured constants a caller can inspect
//! without pulling in an emitter at all. No allocator, no floating-point
//! transcendentals.
//!
//! `alloc` (no_std + alloc): adds [`msl`] -- emission is pure string
//! generation over an already-bound `BoundOp`, so it never needs anything
//! beyond `alloc`'s `String`/`Vec`.
//!
//! `std` (default): adds `std::error::Error` on [`EmitError`], the [`config`]
//! surface, and `backend`, the backend-agnostic `plan_named`/`execute_plan_named`
//! pair, which needs `proxima_tensor::cpu`'s std-gated `Evaluated`/`QuantizedBlock`
//! even with every concrete backend (`cpu`/`metal`/`wgpu-backend`) off.
//!
//! `train` (default-off): adds [`train`], a forward+backward+optimizer step
//! over `proxima-autograd` configured by one `TrainConfig`. It owns
//! `m_<parameter>`/`v_<parameter>` across steps and stops at the dense-gradient
//! case; inference builds never compile it.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
pub mod command_buffer_plan;

#[cfg(feature = "std")]
pub mod backend;
#[cfg(feature = "alloc")]
pub mod bind_plan;
#[cfg(feature = "std")]
pub mod config;
#[cfg(all(feature = "coreml-lane", target_os = "macos"))]
pub mod coreml;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
mod cuda_cached_attention;
#[cfg(feature = "cuda")]
mod cuda_cached_attention_mma;
#[cfg(feature = "cuda")]
mod cuda_cached_attention_tiled;
#[cfg(feature = "cuda")]
pub mod cuda_layout;
#[cfg(feature = "cuda")]
pub mod cuda_encode;
#[cfg(feature = "cuda-driver")]
pub mod cuda_driver;
#[cfg(feature = "cuda-driver")]
mod derived_quantize;
#[cfg(feature = "cuda-driver")]
pub use derived_quantize::ActivationRefusal;
#[cfg(feature = "cuda-driver")]
pub mod kernel_cache_config;
#[cfg(feature = "cuda-driver")]
pub mod cuda_kernel_cache;
#[cfg(feature = "cuda-driver")]
pub mod cuda_ptx_store;
#[cfg(feature = "door")]
pub mod door;
#[cfg(feature = "alloc")]
mod epilogue;
pub mod error;
#[cfg(feature = "alloc")]
mod grid;
#[cfg(feature = "alloc")]
mod identity;
#[cfg(feature = "alloc")]
pub mod resident_identity;
#[cfg(any(feature = "cuda", feature = "metal-grouped-gemm", all(feature = "metal", target_os = "macos"), test))]
mod route_compaction;
#[cfg(feature = "alloc")]
mod lowering;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
#[cfg(proxima_mlx)]
pub mod mlx;
#[cfg(feature = "alloc")]
pub mod encode_render;
#[cfg(feature = "alloc")]
pub mod msl;
#[cfg(feature = "onnx-door")]
pub mod onnx_door;
pub mod sized;
#[cfg(feature = "train")]
pub mod train;
#[cfg(feature = "wgpu-backend")]
pub mod wgpu_driver;
#[cfg(feature = "wgpu-backend")]
pub mod wgsl;
#[cfg(feature = "wgpu-backend")]
pub mod wgsl_layout;
#[cfg(all(
    test,
    feature = "cuda",
    feature = "wgpu-backend",
))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod layout_text_tests;

#[cfg(feature = "cuda")]
pub use cuda::{
    CudaGridSpec, CudaKernel, EmitDevice, WARP_SIZE, emit_cuda, emit_cuda_for_program,
    emit_cuda_on_device, emit_cuda_reusing, emit_cuda_reusing_with_operand_dtypes,
    emit_cuda_with_operand_dtypes, emit_cuda_with_policy,
};
#[cfg(feature = "cuda-driver")]
pub use cuda_driver::CudaF32Arena;
#[cfg(feature = "cuda-driver")]
pub use cuda_driver::CudaPlan;
#[cfg(feature = "cuda-driver")]
pub use cuda_driver::{CudaDeviceSnapshot, CudaDriver, CudaDriverError};
pub use error::EmitError;
#[cfg(feature = "alloc")]
pub use lowering::{OperandDtypes, operand_dtype, operand_dtypes};
#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
pub use metal::{
    CapturedDispatch, execute_plan_timed, flush_gpu_caches, pipeline_cache_keys,
    pipeline_key_audit_counts, pipeline_static_threadgroup_memory, set_capture_step,
    take_captured_dispatches, time_empty_command_buffer_gpu_ns,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal::{
    DispatchType, MathMode, MetalDevice, MetalError, Plan, pipeline_cache_len, checkpoint_mmap_resident_pages,
    discard_checkpoint_mmap_range, discard_checkpoint_mmap_range_immediate, execute, execute_plan,
    execute_plan_named, page_size, plan, plan_named,
};
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
pub use metal::{
    PlacedBuffer, allocate_placed_buffer, allocate_placed_buffer_over,
    execute_plan_named_with_placements, execute_plan_named_with_placements_and_expert_sources,
    execute_plan_named_with_placements_overlapping, execute_plan_with_placements,
    move_placed_buffer_bytes, placed_buffer_len,
    plan_named_with_placed_inputs, plan_named_with_placed_inputs_laid_out, plan_named_with_placed_inputs_scheduled,
    read_placed_buffer, read_placed_buffer_f32_into,
    try_read_placed_buffer_f32, try_write_placed_buffer_f32, write_placed_buffer,
    zero_placed_buffer, zero_placed_buffer_range,
};
#[cfg(all(
    feature = "metal-output-placement",
    feature = "instrument",
    target_os = "macos"
))]
pub use metal::{
    execute_plan_named_with_placements_dispatch_timed, execute_plan_named_with_placements_op_timed,
    execute_plan_with_placements_dispatch_timed, execute_plan_with_placements_op_timed,
    placed_buffer_identity,
};
#[cfg(all(feature = "alloc", feature = "metal-attn-variants"))]
pub use msl::{
    AttentionDecomposition, AttentionDispatchManifest, AttentionKvReuse, AttentionKvStorage, AttentionMmaPrecision,
    AttentionPrefetch, AttentionQueryParallelism, AttentionSimdTopology, AttentionSimdgroupCount,
    AttentionTileHeight, AttentionVariant, emit_with_attention_variant, inspect_attention_variant,
};
#[cfg(feature = "alloc")]
pub use msl::{
    BFLOAT16_BLOCK_BYTES, BFLOAT16_BLOCK_ELEMENTS, Binding, CachedAttentionForm,
    Codec, EmitKnobs, FLOAT16_BLOCK_BYTES, FLOAT16_BLOCK_ELEMENTS, GridSpec, Kernel, PackedOperands,
    Q2K_BLOCK_BYTES, Q2K_BLOCK_ELEMENTS, Q2K_UNPACK_MSL, Q3K_BLOCK_BYTES, Q3K_PAIR_DOT_MSL,
    Q3K_UNPACK_MSL, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q4K_BLOCK_BYTES,
    Q4K_BLOCK_ELEMENTS, Q4K_UNPACK_MSL, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMENTS,
    Q5_1_BLOCK_BYTES, Q5_1_BLOCK_ELEMENTS, Q5K_BLOCK_BYTES, Q5K_UNPACK_MSL,
    Q6K_BLOCK_BYTES, Q6K_UNPACK_MSL, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS, ReductionLiteral, q4_0_unpack_msl,
    q5_0_unpack_msl, q5_1_unpack_msl, q8_0_unpack_msl,
    cached_attention_key_codec_for, cached_attention_kv_codec_for, cached_attention_kv_codecs_for,
    cached_attention_row_tiled_kv_codec_for, cached_attention_row_tiled_reads, context_chunks_for,
    device_row_store_dtype, emit, emit_with_knobs,
};
#[cfg(feature = "wgpu-backend")]
pub use wgpu_driver::{
    WgpuDeviceInfo, WgpuError, WgpuPlan, execute_plan as execute_plan_wgpu,
    execute_plan_named as execute_plan_named_wgpu, plan as plan_wgpu,
    plan_named as plan_named_wgpu, probe as probe_wgpu,
};
#[cfg(feature = "wgpu-backend")]
pub use wgsl::{WORKGROUP_SIZE, WgslCaps, WgslKernel, emit_wgsl, emit_wgsl_with_policy};
#[cfg(test)]
#[cfg(feature = "cuda-driver")]
mod int8_emit_snapshot;
