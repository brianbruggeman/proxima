//! CUDA C text for loading and storing a [`FloatLayout`]: the same integer steps as
//! `msl::layout_render` (which this translates by dialect, never re-derives), with CUDA's own
//! conversion intrinsics where the fields equal `__half` or `__nv_bfloat16`.
//!
//! Text emission only: this box has no CUDA device, so no parity run backs it.

use alloc::format;
use alloc::string::String;

use proxima_numeric::float_layout::FloatLayout;

use crate::backend::cuda::dialect::Cuda;
use crate::cuda::target::Target;
use crate::msl::layout_render::{
    field_store_nan_branch_msl, hardware_type, layout_load_msl, layout_store_msl,
};
use crate::render::{Dialect, HardwareScalar};

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
    let word = |code: &str| reinterpret(&layout, ReinterpretAs::Scalar, code);
    match (
        hardware_type(&layout),
        word("code"),
        word("(unsigned short)code"),
    ) {
        (Some("float"), Some(scalar), _) => format!("\n{signature} {{\n    return {scalar};\n}}\n"),
        (Some("half"), _, Some(scalar)) => {
            format!("\n{signature} {{\n    return __half2float({scalar});\n}}\n")
        }
        (Some(_), _, Some(scalar)) => {
            format!("\n{signature} {{\n    return __bfloat162float({scalar});\n}}\n")
        }
        _ => cuda_dialect(&layout_load_msl(layout, name)),
    }
}

#[derive(Clone, Copy)]
enum ReinterpretAs {
    Scalar,
    Bits,
}

/// The dialect's bit reinterpretation of `value` for the hardware row `layout` equals, in the
/// direction `direction` names.
fn reinterpret(layout: &FloatLayout, direction: ReinterpretAs, value: &str) -> Option<String> {
    let cuda = Cuda {
        target: Target::CUDA,
    };
    let row = HardwareScalar::of(&cuda, layout)?;
    match direction {
        ReinterpretAs::Scalar => cuda.to_scalar(row, value),
        ReinterpretAs::Bits => cuda.to_bits(row, value),
    }
}

/// `__device__ unsigned int name(float value)`: the `layout.width()`-bit code nearest `value`, ties
/// to the even code, overflow and NaN under the layout's specials. The conversion intrinsics emit
/// one canonical NaN, so a NaN input takes the layout's own NaN text and the intrinsic keeps the rest.
#[must_use]
pub fn layout_store_cuda(layout: FloatLayout, name: &str) -> String {
    let signature = format!("__device__ __forceinline__ unsigned int {name}(float value)");
    let bits = |value: &str| reinterpret(&layout, ReinterpretAs::Bits, value);
    let convert = match hardware_type(&layout) {
        Some("float") => {
            if let Some(code) = bits("value") {
                return format!("\n{signature} {{\n    return {code};\n}}\n");
            }
            None
        }
        Some("half") => bits("__float2half_rn(value)"),
        Some(_) => bits("__float2bfloat16_rn(value)"),
        None => None,
    };
    let Some(convert) = convert else {
        return cuda_dialect(&layout_store_msl(layout, name));
    };
    let branch = cuda_dialect(&field_store_nan_branch_msl(&layout));
    format!("\n{signature} {{\n{branch}    return (unsigned int){convert};\n}}\n")
}
