// A transcription of llama.cpp's FP4 block quantizers, line for line, to be `include!`d by tests.
//
// Source: llama.cpp tag b12110 (cd959bae9), `git show b12110:<path>`:
//   ggml/src/ggml-quants.c   best_index_mxfp4, quantize_row_mxfp4_ref (347-379),
//                            quantize_row_nvfp4_ref, dequantize_row_nvfp4 (592)
//   ggml/src/ggml-impl.h     ggml_e8m0_to_fp32_half, ggml_ue4m3_to_fp32, ggml_fp32_to_ue4m3
//   ggml/src/ggml-common.h   kvalues_fp4 (the doubled E2M1 values), block_mxfp4, block_nvfp4
// Names are the C names; the arithmetic is the C arithmetic in `f32`, in the same order.

const LLAMA_KVALUES_FP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

fn llama_e8m0_to_fp32_half(code: u8) -> f32 {
    let bits: u32 = if code < 2 { 0x0020_0000u32 << code } else { u32::from(code - 1) << 23 };
    f32::from_bits(bits)
}

fn llama_best_index_mxfp4(value: f32, scale: f32) -> u8 {
    let mut best_index = 0usize;
    let mut best_error = (f32::from(LLAMA_KVALUES_FP4[0]) * scale - value).abs();
    for (index, level) in LLAMA_KVALUES_FP4.iter().enumerate().skip(1) {
        let error = (f32::from(*level) * scale - value).abs();
        if error < best_error {
            best_index = index;
            best_error = error;
        }
    }
    best_index as u8
}

fn llama_amax(values: &[f32]) -> f32 {
    let mut amax = 0.0f32;
    for value in values {
        if amax < value.abs() {
            amax = value.abs();
        }
    }
    amax
}

/// `quantize_row_mxfp4_ref` for one block of `QK_MXFP4 = 32` values.
pub(crate) fn llama_quantize_block_mxfp4(values: &[f32]) -> [u8; 17] {
    let amax = llama_amax(values);
    let exponent: u8 = if amax > 0.0 { (amax.log2().floor() - 2.0 + 127.0) as i32 as u8 } else { 0 };
    let scale = llama_e8m0_to_fp32_half(exponent);
    let mut block = [0u8; 17];
    block[0] = exponent;
    for index in 0..16 {
        let low = llama_best_index_mxfp4(values[index], scale);
        let high = llama_best_index_mxfp4(values[16 + index], scale);
        block[1 + index] = low | (high << 4);
    }
    block
}

/// `ggml_ue4m3_to_fp32`: the UE4M3 code's value, halved.
fn llama_ue4m3_to_fp32_half(code: u8) -> f32 {
    if code == 0 || code == 0x7f {
        return 0.0;
    }
    let exponent = i32::from((code >> 3) & 0xf);
    let mantissa = f32::from(code & 0x7);
    let power = |shift: i32| f32::from_bits(((shift + 127) as u32) << 23);
    let raw = if exponent == 0 { mantissa * power(-9) } else { (1.0 + mantissa / 8.0) * power(exponent - 7) };
    raw * 0.5
}

/// `ggml_fp32_to_ue4m3`.
fn llama_fp32_to_ue4m3(value: f32) -> u8 {
    // C `!(x > 0.0f)`: a NaN and zero and every negative return code 0
    if value.is_nan() || value <= 0.0 {
        return 0;
    }
    let value = if value > 448.0 { 448.0 } else { value };
    let bits = value.to_bits();
    let fp32_exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let fp32_mantissa = ((bits >> 20) & 0x7) as i32;
    let mut exponent = fp32_exponent + 7;
    if exponent <= 0 {
        let mantissa = ((value * 512.0 + 0.5) as i32).min(7);
        return if mantissa < 1 { 0 } else { mantissa as u8 };
    }
    if exponent >= 15 {
        return 0x7e;
    }
    let round_bit = ((bits >> 19) & 1) as i32;
    let mut mantissa = fp32_mantissa + round_bit;
    if mantissa > 7 {
        mantissa = 0;
        exponent += 1;
        if exponent >= 15 {
            return 0x7e;
        }
    }
    ((exponent << 3) | mantissa) as u8
}

/// `quantize_row_nvfp4_ref` for one block of `QK_NVFP4 = 64` values in four sub-blocks of 16.
pub(crate) fn llama_quantize_block_nvfp4(values: &[f32]) -> [u8; 36] {
    let mut block = [0u8; 36];
    for sub in 0..4 {
        let run = &values[sub * 16..sub * 16 + 16];
        let code = llama_fp32_to_ue4m3(llama_amax(run) / 6.0);
        block[sub] = code;
        let scale = llama_ue4m3_to_fp32_half(code);
        for index in 0..8 {
            let low = llama_best_index_mxfp4(run[index], scale);
            let high = llama_best_index_mxfp4(run[8 + index], scale);
            block[4 + sub * 8 + index] = low | (high << 4);
        }
    }
    block
}

/// `dequantize_row_nvfp4` for one block.
pub(crate) fn llama_dequantize_block_nvfp4(block: &[u8; 36]) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for sub in 0..4 {
        let scale = llama_ue4m3_to_fp32_half(block[sub]);
        for index in 0..8 {
            let byte = block[4 + sub * 8 + index];
            out[sub * 16 + index] = f32::from(LLAMA_KVALUES_FP4[usize::from(byte & 0x0f)]) * scale;
            out[sub * 16 + index + 8] = f32::from(LLAMA_KVALUES_FP4[usize::from(byte >> 4)]) * scale;
        }
    }
    out
}

/// `dequantize_row_mxfp4` for one block (ggml-quants.c): `kvalues * e8m0_to_fp32_half(e)`.
pub(crate) fn llama_dequantize_block_mxfp4(block: &[u8; 17]) -> [f32; 32] {
    let scale = llama_e8m0_to_fp32_half(block[0]);
    let mut out = [0.0f32; 32];
    for index in 0..16 {
        out[index] = f32::from(LLAMA_KVALUES_FP4[usize::from(block[1 + index] & 0x0f)]) * scale;
        out[index + 16] = f32::from(LLAMA_KVALUES_FP4[usize::from(block[1 + index] >> 4)]) * scale;
    }
    out
}
