use proxima_gguf::quant::{bf4_e2m1, bf8_e5m2, bf16, q2_k, q3_k, q4_k, q5_k, scaled_float};
use proxima_primitives::Codec;
use proxima_primitives::float_layout::FloatLayout;
use proxima_tensor::BFloat8;

use crate::codebook;
use crate::error::RecordError;
use crate::incumbent::Incumbent;
use crate::linear;

type BlockEncoder = fn(&[f32], &mut [u8]) -> Result<(), RecordError>;

pub fn encode(codec: Codec, incumbent: Incumbent, rows: &[f32]) -> Result<Vec<u8>, RecordError> {
    let (block_elements, block_bytes) = incumbent.geometry();
    let mut bytes = vec![0_u8; incumbent.byte_len()];
    match block_encoder(codec) {
        Some(encoder) => rows
            .chunks_exact(block_elements)
            .zip(bytes.chunks_exact_mut(block_bytes))
            .try_for_each(|(values, block)| encoder(values, block))?,
        None => whole_row_encoder(codec, rows, &mut bytes)?,
    }
    Ok(bytes)
}

fn whole_row_encoder(codec: Codec, rows: &[f32], bytes: &mut [u8]) -> Result<(), RecordError> {
    match codec {
        Codec::BFloat16 => {
            bf16::quantize(rows, bytes).map_err(|error| RecordError::quant("encode", codec, error))
        }
        Codec::Q2K => {
            q2_k::quantize(rows, bytes).map_err(|error| RecordError::quant("encode", codec, error))
        }
        Codec::Q3K => {
            q3_k::quantize(rows, bytes).map_err(|error| RecordError::quant("encode", codec, error))
        }
        Codec::Q4K => {
            q4_k::quantize(rows, bytes).map_err(|error| RecordError::quant("encode", codec, error))
        }
        Codec::Q5K => {
            q5_k::quantize(rows, bytes).map_err(|error| RecordError::quant("encode", codec, error))
        }
        Codec::Q8K => proxima_tensor::cpu::quantize_row_q8k(rows, bytes)
            .map_err(|error| RecordError::quant("encode", codec, error)),
        other => Err(RecordError::quant("encode", other, "no encoder registered")),
    }
}

fn block_encoder(codec: Codec) -> Option<BlockEncoder> {
    Some(match codec {
        Codec::BFloat8 => bfloat8_block,
        Codec::Bf8E5M2 => bf8_e5m2_block,
        Codec::Bf4E2M1 => bf4_e2m1_block,
        Codec::Fp6E3M2 => fp6_block,
        Codec::Q4_1 => linear::q4_1_block,
        Codec::Q5_0 => linear::q5_0_block,
        Codec::Q5_1 => linear::q5_1_block,
        Codec::Q8_1 => linear::q8_1_block,
        Codec::Q1_0 => linear::q1_0_block,
        Codec::Q2_0 => linear::q2_0_block,
        Codec::Iq4Nl => codebook::iq4_nl_block,
        Codec::Iq2Xs => codebook::iq2_xs_block,
        Codec::Iq3Xxs => codebook::iq3_xxs_block,
        _ => return None,
    })
}

fn bfloat8_block(values: &[f32], block: &mut [u8]) -> Result<(), RecordError> {
    block[0] = BFloat8::from_f32(values[0]).to_bits();
    Ok(())
}

fn bf8_e5m2_block(values: &[f32], block: &mut [u8]) -> Result<(), RecordError> {
    block[0] = bf8_e5m2::encode(values[0]);
    Ok(())
}

fn bf4_e2m1_block(values: &[f32], block: &mut [u8]) -> Result<(), RecordError> {
    block[0] = bf4_e2m1::pack_pair(values[0], values[1])
        .map_err(|error| RecordError::quant("encode", Codec::Bf4E2M1, error))?;
    Ok(())
}

fn fp6_block(values: &[f32], block: &mut [u8]) -> Result<(), RecordError> {
    let failed = |reason| RecordError::quant("encode", Codec::Fp6E3M2, reason);
    let values = <&[f32; scaled_float::BLOCK_ELEMENTS]>::try_from(values)
        .map_err(|error| failed(format!("{error:?}")))?;
    let block = <&mut [u8; scaled_float::BLOCK_BYTES]>::try_from(block)
        .map_err(|error| failed(format!("{error:?}")))?;
    scaled_float::quantize_block(values, FloatLayout::E3M2, block)
        .map_err(|error| failed(format!("{error:?}")))
}
