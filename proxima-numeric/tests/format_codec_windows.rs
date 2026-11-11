//! `FormatCodec` decode values, which the crate's own `#![forbid(unsafe_code)]` unit tests cannot read
//! back: reading a decoded `MaybeUninit<f32>` needs `assume_init`. The inputs are a deterministic
//! pseudo-random stream of full-range f32 bit patterns (so NaN, infinities and subnormals occur) with a
//! signalling NaN written over the first element, the case a vector-converting copy quiets.

// tests state their failures through expect messages
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::mem::MaybeUninit;

use proxima_numeric::{BlockLayout, FloatLayout, FormatCodec};

/// the element count up to which an identity decode copies inline, in `format_codec.rs`
const IDENTITY_INLINE_ELEMENTS: usize = 8192;
const SIGNALLING_NAN_BITS: u32 = 0x7fa0_1234;
const WINDOW_TAIL: usize = 17;
const TAIL_BYTE: u8 = 0xAB;

fn kv_values(count: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32;
    let mut values: Vec<f32> = (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from_bits(state ^ (state << 13))
        })
        .collect();
    values[0] = f32::from_bits(SIGNALLING_NAN_BITS);
    values
}

fn respelled_f16() -> FloatLayout {
    FloatLayout { bias_override: Some(FloatLayout::F16.bias()), ..FloatLayout::F16 }
}

fn route_codecs() -> Vec<(&'static str, FormatCodec)> {
    vec![
        ("identity", FormatCodec::scalar(&FloatLayout::BF32).expect("the source layout resolves")),
        ("named scalar", FormatCodec::scalar(&FloatLayout::BF16).expect("bf16 resolves")),
        ("layout as data", FormatCodec::scalar(&respelled_f16()).expect("a respelled f16 resolves")),
        ("block", FormatCodec::block(&BlockLayout::Q4_0).expect("q4_0 resolves")),
    ]
}

fn decode_bits(codec: &FormatCodec, data: &[u8], count: usize) -> Vec<u32> {
    let mut slots = vec![MaybeUninit::<f32>::uninit(); count];
    codec.decode(data, &mut slots).expect("slots are whole blocks and data holds their range");
    // SAFETY: decode initialises every slot when it returns Ok.
    slots.iter().map(|slot| unsafe { slot.assume_init() }.to_bits()).collect()
}

#[test]
fn an_identity_decode_returns_every_input_bit_pattern_on_both_sides_of_the_inline_bound() {
    let codec = FormatCodec::scalar(&FloatLayout::BF32).expect("the source layout resolves");
    for count in [256, 512, IDENTITY_INLINE_ELEMENTS, IDENTITY_INLINE_ELEMENTS + 1] {
        let values = kv_values(count);
        let mut packed = vec![0u8; count * 4];
        codec.encode(&values, &mut packed).expect("packed holds four bytes per value");

        let decoded = decode_bits(&codec, &packed, count);

        let expected: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
        assert_eq!(decoded[0], SIGNALLING_NAN_BITS, "{count} elements: the signalling NaN payload");
        assert_eq!(decoded, expected, "{count} elements");
    }
}

#[test]
fn a_longer_window_decodes_to_the_same_values_as_the_exact_buffer_on_every_route() {
    let values = kv_values(256);
    for (route, codec) in route_codecs() {
        let packed_len = codec.packed_len(values.len()).expect("whole blocks");
        let mut exact = vec![0u8; packed_len];
        codec.encode(&values, &mut exact).expect("the buffer is the packed range");
        let mut window = vec![TAIL_BYTE; packed_len + WINDOW_TAIL];
        window[..packed_len].copy_from_slice(&exact);

        let from_exact = decode_bits(&codec, &exact, values.len());
        let from_window = decode_bits(&codec, &window, values.len());

        assert_eq!(from_window, from_exact, "{route} decode values");
    }
}
