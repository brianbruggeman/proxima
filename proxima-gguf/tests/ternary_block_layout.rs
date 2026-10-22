use proxima_gguf::GgmlType;
use proxima_gguf::quant::dispatch::dequantize;

#[test]
fn tq2_uses_the_shared_value_scale_and_bit_plane_path() {
    let mut block = [0_u8; 66];
    block[0] = 0b11_10_01_00;
    block[64..66].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    let mut output = [0.0_f32; 256];

    dequantize(GgmlType::Tq20, &block, &mut output).expect("TQ2_0 block decodes");

    assert_eq!(output[0], -2.0);
    assert_eq!(output[32], 0.0);
    assert_eq!(output[64], 2.0);
    assert_eq!(output[96], 4.0);
    assert!(output[1..32].iter().all(|value| *value == -2.0));
    assert!(output[128..].iter().all(|value| *value == -2.0));
}

#[test]
fn tq1_matches_the_hand_derived_radix_three_regions() {
    let mut block = [0_u8; 54];
    block[0] = 85;
    block[32] = 85;
    block[48] = 83;
    block[52..54].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    let mut output = [0.0_f32; 256];

    dequantize(GgmlType::Tq10, &block, &mut output).expect("TQ1_0 block decodes");

    let mut expected = [-2.0_f32; 256];
    expected[32] = 2.0;
    expected[64] = 2.0;
    expected[96] = 2.0;
    expected[128] = 2.0;
    expected[176] = 2.0;
    expected[192] = 2.0;
    expected[208] = 2.0;
    expected[224] = 2.0;
    expected[244] = 2.0;
    expected[248] = 2.0;
    expected[252] = 2.0;
    assert_eq!(output, expected);
}
