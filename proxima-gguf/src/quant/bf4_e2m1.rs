//! Scalar E2M1 brain-float conversion and low-nibble-first byte packing.

use crate::quant::QuantError;

const CODEC: &str = "bf4_e2m1";
const LEVELS: [(f32, u8); 8] = [
    (0.0, 0),
    (0.5, 1),
    (1.0, 2),
    (1.5, 3),
    (2.0, 4),
    (3.0, 5),
    (4.0, 6),
    (6.0, 7),
];

/// Rounds one finite value to the selected E2M1 grid, saturating at ±6.
pub fn encode(value: f32) -> Result<u8, QuantError> {
    if !value.is_finite() {
        return Err(QuantError::NonFiniteInput { codec: CODEC });
    }

    let sign = if value.is_sign_negative() { 0x08 } else { 0 };
    let magnitude = value.abs().min(6.0);
    let mut best_code = 0;
    let mut best_distance = f32::INFINITY;
    for (level, code) in LEVELS {
        let distance = (magnitude - level).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0) {
            best_code = code;
            best_distance = distance;
        }
    }
    Ok(sign | best_code)
}

/// Decodes one E2M1 nibble.
#[must_use]
pub fn decode(encoded: u8) -> f32 {
    let nibble = encoded & 0x0f;
    let sign = if nibble & 0x08 == 0 { 1.0 } else { -1.0 };
    let exponent = (nibble >> 1) & 0x03;
    let fraction = nibble & 0x01;
    if exponent == 0 {
        return sign * f32::from(fraction) * 0.5;
    }
    let significand = 1.0 + f32::from(fraction) * 0.5;
    let scale = match exponent {
        1 => 1.0,
        2 => 2.0,
        _ => 4.0,
    };
    sign * significand * scale
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
