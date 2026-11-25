//! Backend-neutral rendering of numeric layouts.
//!
//! The integer steps that decode and encode a [`proxima_numeric::float_layout::FloatLayout`] are
//! one algorithm; a [`Dialect`] spells them for a target language and a [`HardwareScalar`] names
//! the wide scalar the steps compute in. `msl::dialect::Msl` is the first dialect.
//!
//! Teaching pointer: `proxima_numeric::float_layout_codec::{decode, encode}` is the CPU algorithm
//! these functions render step for step.

mod dialect;
mod hardware;
mod layout;
mod refusal;

pub use dialect::{Dialect, Intrinsic};
pub(crate) use hardware::HARDWARE_COUNT;
pub use hardware::HardwareScalar;
#[cfg(feature = "wgpu-backend")]
pub(crate) use layout::{
    exponent_ones, infinity_code, mantissa_mask, max_normal_code, overflow_code, quiet_bit,
    reserves_all_ones, sign_shift,
};
pub use layout::{load, store, store_nan_branch, unpack};
pub use refusal::Refusal;
