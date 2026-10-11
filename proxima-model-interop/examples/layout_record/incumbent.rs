use proxima_gguf::GgmlType;
use proxima_gguf::quant::{bf4_e2m1, bf8_e5m2, dispatch, scaled_float};
use proxima_primitives::Codec;
use proxima_primitives::float_layout::FloatLayout;
use proxima_tensor::BFloat8;

use crate::error::RecordError;

pub const ELEMENTS: usize = 4096;

#[derive(Clone, Copy, Debug)]
pub enum Incumbent {
    Ggml(GgmlType),
    Bfloat8,
    Bf8E5M2,
    Bf4E2M1,
    Fp6E3M2,
}

pub fn incumbent(codec: Codec) -> Option<Incumbent> {
    match codec {
        Codec::BFloat8 => Some(Incumbent::Bfloat8),
        Codec::Bf8E5M2 => Some(Incumbent::Bf8E5M2),
        Codec::Bf4E2M1 => Some(Incumbent::Bf4E2M1),
        Codec::Fp6E3M2 => Some(Incumbent::Fp6E3M2),
        other => dispatch_type(other).map(Incumbent::Ggml),
    }
}

fn dispatch_type(codec: Codec) -> Option<GgmlType> {
    Some(match codec {
        Codec::Float16 => GgmlType::F16,
        Codec::BFloat16 => GgmlType::Bf16,
        Codec::Q2K => GgmlType::Q2_K,
        Codec::Q3K => GgmlType::Q3_K,
        Codec::Q4K => GgmlType::Q4_K,
        Codec::Q5K => GgmlType::Q5_K,
        Codec::Q6K => GgmlType::Q6_K,
        Codec::Q8K => GgmlType::Q8_K,
        Codec::Q8_0 => GgmlType::Q8_0,
        Codec::Q8_1 => GgmlType::Q8_1,
        Codec::Q4_0 => GgmlType::Q4_0,
        Codec::Q4_1 => GgmlType::Q4_1,
        Codec::Q5_0 => GgmlType::Q5_0,
        Codec::Q5_1 => GgmlType::Q5_1,
        Codec::Q1_0 => GgmlType::Q1_0,
        Codec::Q2_0 => GgmlType::Q2_0,
        Codec::Iq4Nl => GgmlType::Iq4Nl,
        Codec::Iq2Xs => GgmlType::Iq2Xs,
        Codec::Iq3Xxs => GgmlType::Iq3Xxs,
        _ => return None,
    })
}

impl Incumbent {
    pub fn geometry(self) -> (usize, usize) {
        match self {
            Self::Bfloat8 | Self::Bf8E5M2 => (1, 1),
            Self::Bf4E2M1 => (2, 1),
            Self::Fp6E3M2 => Self::ggml_geometry(GgmlType::Fp6E3M2),
            Self::Ggml(kind) => Self::ggml_geometry(kind),
        }
    }

    fn ggml_geometry(kind: GgmlType) -> (usize, usize) {
        let layout = kind.block_layout();
        (layout.block_elements as usize, layout.block_bytes as usize)
    }

    pub fn byte_len(self) -> usize {
        let (block_elements, block_bytes) = self.geometry();
        ELEMENTS / block_elements * block_bytes
    }

    pub fn decode(
        self,
        codec: Codec,
        blocks: &[u8],
        output: &mut [f32],
    ) -> Result<(), RecordError> {
        match self {
            Self::Ggml(kind) => dispatch::dequantize(kind, blocks, output)
                .map_err(|error| RecordError::quant("decode", codec, error)),
            Self::Bfloat8 => decode_each(blocks, output, |byte| BFloat8::from_bits(byte).to_f32()),
            Self::Bf8E5M2 => decode_each(blocks, output, bf8_e5m2::decode),
            Self::Bf4E2M1 => decode_pairs(blocks, output),
            Self::Fp6E3M2 => decode_fp6(blocks, output)
                .map_err(|error| RecordError::quant("decode", codec, error)),
        }
    }
}

fn decode_each(
    blocks: &[u8],
    output: &mut [f32],
    decode: fn(u8) -> f32,
) -> Result<(), RecordError> {
    blocks
        .iter()
        .zip(output)
        .for_each(|(byte, slot)| *slot = decode(*byte));
    Ok(())
}

fn decode_pairs(blocks: &[u8], output: &mut [f32]) -> Result<(), RecordError> {
    blocks
        .iter()
        .zip(output.as_chunks_mut::<2>().0)
        .for_each(|(byte, pair)| *pair = bf4_e2m1::unpack_pair(*byte));
    Ok(())
}

fn decode_fp6(blocks: &[u8], output: &mut [f32]) -> Result<(), String> {
    let (packed, _) = blocks.as_chunks::<{ scaled_float::BLOCK_BYTES }>();
    let (values, _) = output.as_chunks_mut::<{ scaled_float::BLOCK_ELEMENTS }>();
    packed
        .iter()
        .zip(values)
        .try_for_each(|(block, block_values)| {
            scaled_float::dequantize_block(block, FloatLayout::E3M2, block_values)
        })
        .map_err(|error| format!("{error:?}"))
}
