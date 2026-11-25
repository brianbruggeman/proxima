//! `device_scalar` is the one decision behind both the emitted type token and the device byte
//! width, so every dtype proxima-numeric names must agree across the two.

use alloc::vec::Vec;

use moe_test_json as serde_json;
use proxima_numeric::DType;
use proxima_numeric::float_layout::NAMED_LAYOUTS;
use proxima_tensor::NodeId;

use super::target::Target;
use super::{DeviceClass, device_scalar, type_token};
use crate::error::EmitError;
use crate::msl::layout_render::hardware_word;

/// Every named layout plus the one non-numeric dtype, obtained through its serde name so the
/// list comes from proxima-numeric rather than a hand list.
fn every_dtype() -> Vec<DType> {
    let mut dtypes: Vec<DType> = NAMED_LAYOUTS.iter().map(|(layout, _)| DType::Layout(*layout)).collect();
    let non_numeric: DType = serde_json::from_str("\"bool\"").expect("proxima-numeric spells the non-numeric dtype");
    dtypes.push(non_numeric);
    dtypes
}

#[test]
fn device_scalar_agrees_with_type_token_for_every_named_dtype() {
    let dtypes = every_dtype();
    assert!(dtypes.len() > 2, "proxima-numeric names several layouts");
    let (mut wide, mut narrow) = (0, 0);
    for target in [Target::CUDA, Target::HIP] {
        let mut word_tokens = Vec::new();
        for dtype in &dtypes {
            let token = type_token(target, NodeId(0), *dtype);
            let Some(scalar) = device_scalar(*dtype) else {
                wide += 1;
                assert!(
                    matches!(token, Err(EmitError::UnsupportedDType { .. })),
                    "{dtype:?} has no device scalar, so the emitter must refuse it"
                );
                continue;
            };
            narrow += 1;
            let token = token.expect("a dtype with a device scalar has a token");
            match scalar.class {
                DeviceClass::Half => {
                    assert_eq!(token, "__half", "{dtype:?}");
                    assert_eq!(scalar.bytes, size_of::<u16>(), "{dtype:?}");
                }
                DeviceClass::Bfloat => {
                    assert_eq!(token, target.bfloat_type, "{dtype:?}");
                    assert_eq!(scalar.bytes, size_of::<u16>(), "{dtype:?}");
                }
                DeviceClass::Word => {
                    let row = hardware_word().expect("the hardware table has a word row");
                    assert_eq!(token, row.scalar, "{dtype:?} word token is the word row's scalar");
                    assert_eq!(scalar.bytes, row.bytes as usize, "{dtype:?} word bytes are the word row's");
                    word_tokens.push(token);
                    assert_eq!(scalar.bytes, size_of::<u32>(), "{dtype:?}");
                }
            }
        }
        word_tokens.dedup();
        assert_eq!(word_tokens.len(), 1, "every word-class dtype shares one token");
    }
    assert!(wide > 0, "the named layouts include one wider than 32 bits");
    assert!(narrow > 0, "the named layouts include one a device scalar holds");
}

#[test]
fn device_scalar_bytes_hold_the_logical_element_for_every_named_dtype() {
    for dtype in every_dtype() {
        let Some(scalar) = device_scalar(dtype) else { continue };
        assert!(scalar.bytes >= dtype.size_bytes(), "{dtype:?} device element must hold the logical one");
        if scalar.class != DeviceClass::Word {
            assert_eq!(scalar.bytes, dtype.size_bytes(), "{dtype:?} narrow device scalar is stored as-is");
        }
    }
}
