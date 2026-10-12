use proxima_gguf::quant::float_layout::decode;
use proxima_primitives::float_layout::FloatLayout;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Comparison {
    pub mismatches: usize,
    pub max_ulp: u64,
}

fn ordered(code: u32, layout: FloatLayout) -> i64 {
    let sign_shift = u32::from(layout.exponent_bits) + u32::from(layout.mantissa_bits);
    let magnitude = i64::from(code & ((1u32 << sign_shift) - 1));
    if (code >> sign_shift) & 1 == 1 {
        -magnitude
    } else {
        magnitude
    }
}

pub fn compare_codes(layout: FloatLayout, expected: &[u32], observed: &[u32]) -> Comparison {
    assert_eq!(
        expected.len(),
        observed.len(),
        "compared sequences differ in length"
    );
    let mut comparison = Comparison {
        mismatches: 0,
        max_ulp: 0,
    };
    for (&want, &got) in expected.iter().zip(observed) {
        let both_nan = decode(want, layout).is_nan() && decode(got, layout).is_nan();
        if want == got || both_nan {
            continue;
        }
        comparison.mismatches += 1;
        let distance = ordered(want, layout).abs_diff(ordered(got, layout));
        comparison.max_ulp = comparison.max_ulp.max(distance);
    }
    comparison
}

pub fn compare_bits(expected: &[f32], observed: &[f32]) -> usize {
    assert_eq!(
        expected.len(),
        observed.len(),
        "compared sequences differ in length"
    );
    expected
        .iter()
        .zip(observed)
        .filter(|(want, got)| want.to_bits() != got.to_bits())
        .count()
}
