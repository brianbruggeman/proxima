//! Scalar E2M1 brain-float conversion and low-nibble-first byte packing.
//!
//! Teaching pointer: wraps the generic [`float_layout`] codec with the E2M1 layout; only the
//! non-finite rejection lives here.

use proxima_primitives::float_layout::FloatLayout;

use crate::quant::QuantError;
use crate::quant::float_layout;

const CODEC: &str = "bf4_e2m1";

/// Rounds one finite value to the selected E2M1 grid, saturating at ±6.
pub fn encode(value: f32) -> Result<u8, QuantError> {
    if !value.is_finite() {
        return Err(QuantError::NonFiniteInput { codec: CODEC });
    }
    Ok(float_layout::encode(value, FloatLayout::E2M1) as u8)
}

/// Decodes one E2M1 nibble.
#[must_use]
pub fn decode(encoded: u8) -> f32 {
    float_layout::decode(u32::from(encoded & 0x0f), FloatLayout::E2M1)
}

/// Packs `first` into the low nibble and `second` into the high nibble.
pub fn pack_pair(first: f32, second: f32) -> Result<u8, QuantError> {
    let low = encode(first)?;
    let high = encode(second)?;
    Ok(low | (high << 4))
}

/// Decodes a byte as `[low_nibble_value, high_nibble_value]`.
#[must_use]
pub fn unpack_pair(packed: u8) -> [f32; 2] {
    [decode(packed & 0x0f), decode(packed >> 4)]
}
