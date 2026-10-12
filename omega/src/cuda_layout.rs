//! CUDA C text for loading and storing a [`FloatLayout`]: the same integer steps as
//! `msl::layout_render` (which this translates by dialect, never re-derives), with CUDA's own
//! conversion intrinsics where the fields equal `__half` or `__nv_bfloat16`.
//!
//! Text emission only: this box has no CUDA device, so no parity run backs it.

use alloc::format;
use alloc::string::String;

use proxima_primitives::float_layout::FloatLayout;

use crate::msl::layout_render::{hardware_type, layout_load_msl, layout_store_msl};

fn cuda_dialect(msl: &str) -> String {
    msl.replace("uint", "unsigned int")
        .replace("static inline ", "__device__ __forceinline__ ")
        .replace("as_type<float>", "__uint_as_float")
        .replace("as_type<unsigned int>", "__float_as_uint")
        .replace("clz(", "__clz(")
}

/// `__device__ float name(unsigned int code)`: the value of the `layout.width()`-bit `code`.
#[must_use]
pub fn layout_load_cuda(layout: FloatLayout, name: &str) -> String {
    let signature = format!("__device__ __forceinline__ float {name}(unsigned int code)");
    match hardware_type(&layout) {
        Some("float") => format!("\n{signature} {{\n    return __uint_as_float(code);\n}}\n"),
        Some("half") => format!(
            "\n{signature} {{\n    return __half2float(__ushort_as_half((unsigned short)code));\n}}\n"
        ),
        Some(_) => format!(
            "\n{signature} {{\n    return __bfloat162float(__ushort_as_bfloat16((unsigned short)code));\n}}\n"
        ),
        None => cuda_dialect(&layout_load_msl(layout, name)),
    }
}

/// `__device__ unsigned int name(float value)`: the `layout.width()`-bit code nearest `value`, ties
/// to the even code, overflow and NaN under the layout's specials.
#[must_use]
pub fn layout_store_cuda(layout: FloatLayout, name: &str) -> String {
    let signature = format!("__device__ __forceinline__ unsigned int {name}(float value)");
    match hardware_type(&layout) {
        Some("float") => format!("\n{signature} {{\n    return __float_as_uint(value);\n}}\n"),
        Some("half") => format!(
            "\n{signature} {{\n    return (unsigned int)__half_as_ushort(__float2half_rn(value));\n}}\n"
        ),
        Some(_) => format!(
            "\n{signature} {{\n    return (unsigned int)__bfloat16_as_ushort(__float2bfloat16_rn(value));\n}}\n"
        ),
        None => cuda_dialect(&layout_store_msl(layout, name)),
    }
}
