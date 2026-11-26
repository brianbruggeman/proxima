//! CUDA C text for loading and storing a [`FloatLayout`]: the backend-neutral
//! [`crate::render::load`] / [`crate::render::store`] steps spelled by the
//! [`Cuda`](crate::backend::cuda::dialect::Cuda) dialect, so CUDA and HIP share the one path every
//! other dialect takes and only the target's bfloat name differs.
//!
//! Text emission only: this box has no CUDA device, so no parity run backs it.

use alloc::string::String;

use proxima_numeric::float_layout::FloatLayout;

use crate::backend::cuda::dialect::Cuda;
use crate::cuda::target::Target;
use crate::render::{self, HardwareScalar, Refusal};

/// `__device__ float name(unsigned int code)`: the value of the `layout.width()`-bit `code`.
///
/// # Errors
/// [`Refusal`] when the dialect cannot spell the layout; see [`render::load`].
#[must_use = "the refusal names why the layout could not be rendered"]
pub fn layout_load_cuda(
    target: Target,
    layout: FloatLayout,
    name: &str,
) -> Result<String, Refusal> {
    let cuda = Cuda { target };
    let wide = HardwareScalar::widest(&cuda).ok_or(Refusal::NoWideScalar)?;
    render::load(&cuda, wide, layout, name)
}

/// `__device__ unsigned int name(float value)`: the `layout.width()`-bit code nearest `value`, ties
/// to the even code, overflow and NaN under the layout's specials.
///
/// # Errors
/// [`Refusal`] when the dialect cannot spell the layout; see [`render::store`].
#[must_use = "the refusal names why the layout could not be rendered"]
pub fn layout_store_cuda(
    target: Target,
    layout: FloatLayout,
    name: &str,
) -> Result<String, Refusal> {
    let cuda = Cuda { target };
    let wide = HardwareScalar::widest(&cuda).ok_or(Refusal::NoWideScalar)?;
    render::store(&cuda, wide, layout, name)
}
