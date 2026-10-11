//! Scalar E5M2 brain-float conversion used by the low-precision training path.
//!
//! Teaching pointer: a thin wrapper over the generic [`float_layout`] codec with the saturating
//! E5M2 layout, so infinities never reach stored weights.

use proxima_primitives::float_layout::FloatLayout;

use crate::quant::float_layout;

/// The E5M2 encoding for one value.
#[must_use]
pub fn encode(value: f32) -> u8 {
    float_layout::encode(value, FloatLayout::E5M2_SATURATING) as u8
}

/// Decodes one E5M2 byte, including its infinity and NaN patterns.
#[must_use]
pub fn decode(encoded: u8) -> f32 {
    float_layout::decode(u32::from(encoded), FloatLayout::E5M2_SATURATING)
}
