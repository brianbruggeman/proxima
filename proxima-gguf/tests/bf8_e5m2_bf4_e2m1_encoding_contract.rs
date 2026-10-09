use proxima_gguf::quant::{QuantError, bf4_e2m1, bf8_e5m2};

#[test]
fn bf8_e5m2_bf4_e2m1_encoding_contract() {
    let bf8_subnormals = [
        0.0,
        2.0_f32.powi(-16),
        2.0_f32.powi(-15),
        3.0 * 2.0_f32.powi(-16),
    ];
    for encoded in 0_u8..=u8::MAX {
        let expected = match encoded {
            0x7c => f32::INFINITY,
            0xfc => f32::NEG_INFINITY,
            0x7d..=0x7f | 0xfd..=0xff => f32::NAN,
            _ => {
                let sign = if encoded & 0x80 == 0 { 1.0 } else { -1.0 };
                let exponent = usize::from((encoded >> 2) & 0x1f);
                let fraction = encoded & 0x03;
                if exponent == 0 {
                    sign * bf8_subnormals[usize::from(fraction)]
                } else {
                    let exponent_value = i32::from(exponent as u8) - 15;
                    let significand = [1.0, 1.25, 1.5, 1.75][usize::from(fraction)];
                    sign * significand * 2.0_f32.powi(exponent_value)
                }
            }
        };
        let actual = bf8_e5m2::decode(encoded);
        if expected.is_nan() {
            assert!(
                actual.is_nan(),
                "E5M2 byte {encoded:#04x} must decode as NaN"
            );
        } else {
            assert_eq!(actual, expected, "E5M2 byte {encoded:#04x}");
        }
    }

    let bf4_expected = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    for (encoded, expected) in bf4_expected.into_iter().enumerate() {
        assert_eq!(
            bf4_e2m1::decode(encoded as u8),
            expected,
            "E2M1 nibble {encoded:#x}"
        );
    }

    assert_eq!(bf8_e5m2::encode(1.125), 0x3c);
    assert_eq!(bf8_e5m2::encode(-1.125), 0xbc);
    assert_eq!(bf8_e5m2::encode(f32::MAX), 0x7b);
    assert_eq!(bf8_e5m2::encode(f32::INFINITY), 0x7b);
    assert_eq!(bf8_e5m2::encode(f32::NEG_INFINITY), 0xfb);
    assert_eq!(bf8_e5m2::encode(f32::NAN), 0x7e);

    assert_eq!(bf4_e2m1::encode(1.25), Ok(0x2));
    assert_eq!(bf4_e2m1::encode(-1.25), Ok(0xa));
    assert_eq!(bf4_e2m1::encode(99.0), Ok(0x7));
    assert_eq!(bf4_e2m1::encode(-99.0), Ok(0xf));
    assert_eq!(
        bf4_e2m1::encode(f32::INFINITY),
        Err(QuantError::NonFiniteInput { codec: "bf4_e2m1" })
    );
    assert_eq!(
        bf4_e2m1::encode(f32::NAN),
        Err(QuantError::NonFiniteInput { codec: "bf4_e2m1" })
    );

    let packed = bf4_e2m1::pack_pair(1.0, 0.5).expect("finite E2M1 values");
    assert_eq!(packed, 0x12);
    assert_eq!(bf4_e2m1::unpack_pair(packed), [1.0, 0.5]);
}
