use std::io;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    F32,
    Bf16,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
        }
    }

    pub fn round(self, value: f32) -> io::Result<f32> {
        if !value.is_finite() {
            return Err(io::Error::other(
                "oracle arithmetic produced a nonfinite value",
            ));
        }
        match self {
            Self::F32 => Ok(value),
            Self::Bf16 => {
                let bits = value.to_bits();
                let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
                let output = f32::from_bits(rounded);
                if !output.is_finite() {
                    return Err(io::Error::other("bf16 rounding overflowed to nonfinite"));
                }
                Ok(output)
            }
        }
    }
}

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        transpose_left: i32,
        transpose_right: i32,
        rows: i32,
        columns: i32,
        reduction: i32,
        alpha: f32,
        left: *const f32,
        left_stride: i32,
        right: *const f32,
        right_stride: i32,
        beta: f32,
        output: *mut f32,
        output_stride: i32,
    );
}

pub fn decode(bytes: &[u8]) -> io::Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(io::Error::other("bf16 byte count is odd"));
    }
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let word = u16::from_le_bytes(*pair);
            let value = f32::from_bits(u32::from(word) << 16);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(io::Error::other("checkpoint bf16 value is nonfinite"))
            }
        })
        .collect()
}

pub fn linear(
    input: &[f32],
    weights: &[f32],
    positions: usize,
    input_width: usize,
    output_width: usize,
    mode: Mode,
) -> io::Result<Vec<f32>> {
    if positions == 0 || input_width == 0 || output_width == 0 {
        return Err(io::Error::other("linear dimensions must be positive"));
    }
    if input.len()
        != positions
            .checked_mul(input_width)
            .ok_or_else(|| io::Error::other("input shape overflow"))?
        || weights.len()
            != output_width
                .checked_mul(input_width)
                .ok_or_else(|| io::Error::other("weight shape overflow"))?
    {
        return Err(io::Error::other(
            "linear input or physical out/in weight dimensions differ",
        ));
    }
    let mut output = accelerated_product(input, weights, positions, input_width, output_width)?;
    for value in &mut output {
        *value = mode.round(*value)?;
    }
    Ok(output)
}

#[cfg(target_os = "macos")]
fn accelerated_product(
    input: &[f32],
    weights: &[f32],
    positions: usize,
    input_width: usize,
    output_width: usize,
) -> io::Result<Vec<f32>> {
    let rows = i32::try_from(positions).map_err(io::Error::other)?;
    let columns = i32::try_from(output_width).map_err(io::Error::other)?;
    let reduction = i32::try_from(input_width).map_err(io::Error::other)?;
    let mut output = vec![
        0.0;
        positions
            .checked_mul(output_width)
            .ok_or_else(|| io::Error::other("output shape overflow"))?
    ];
    // safety: validated extents and a fresh output make these row-major buffers non-aliasing.
    unsafe {
        cblas_sgemm(
            101,
            111,
            112,
            rows,
            columns,
            reduction,
            1.0,
            input.as_ptr(),
            reduction,
            weights.as_ptr(),
            reduction,
            0.0,
            output.as_mut_ptr(),
            columns,
        );
    }
    Ok(output)
}

#[cfg(not(target_os = "macos"))]
fn accelerated_product(
    _input: &[f32],
    _weights: &[f32],
    _positions: usize,
    _input_width: usize,
    _output_width: usize,
) -> io::Result<Vec<f32>> {
    Err(io::Error::other("oracle requires native macos Accelerate"))
}

pub fn norm(input: &[f32], scale: &[f32], epsilon: f32, mode: Mode) -> io::Result<Vec<f32>> {
    let width = scale.len();
    if width == 0 || !input.len().is_multiple_of(width) || epsilon < 0.0 {
        return Err(io::Error::other(
            "normalization dimensions or epsilon differ",
        ));
    }
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(width) {
        let mut square_sum = 0.0f32;
        for value in row {
            square_sum += *value * *value;
        }
        let variance = square_sum / width as f32;
        let inverse = 1.0 / (variance + epsilon).sqrt();
        for (value, learned_scale) in row.iter().zip(scale) {
            let normalized = mode.round(*value * inverse)?;
            output.push(mode.round(normalized * *learned_scale)?);
        }
    }
    Ok(output)
}

pub fn rope_tables(
    positions: usize,
    head_width: usize,
    base: f32,
    mode: Mode,
) -> io::Result<(Vec<f32>, Vec<f32>)> {
    if head_width == 0 || !head_width.is_multiple_of(2) || base <= 0.0 {
        return Err(io::Error::other("rotary dimensions or base differ"));
    }
    let mut cosine = Vec::with_capacity(positions * head_width);
    let mut sine = Vec::with_capacity(positions * head_width);
    for position in 0..positions {
        for component in 0..head_width {
            let pair = component % (head_width / 2);
            let inverse = 1.0 / base.powf((2 * pair) as f32 / head_width as f32);
            let angle = position as f32 * inverse;
            cosine.push(mode.round(angle.cos())?);
            sine.push(mode.round(angle.sin())?);
        }
    }
    Ok((cosine, sine))
}

pub fn rope(
    input: &[f32],
    cosine: &[f32],
    sine: &[f32],
    positions: usize,
    heads: usize,
    head_width: usize,
    mode: Mode,
) -> io::Result<Vec<f32>> {
    let width = heads * head_width;
    if head_width == 0
        || !head_width.is_multiple_of(2)
        || input.len() != positions * width
        || cosine.len() != positions * head_width
        || sine.len() != cosine.len()
    {
        return Err(io::Error::other("rotary input/table dimensions differ"));
    }
    let mut output = vec![0.0; input.len()];
    let half = head_width / 2;
    for position in 0..positions {
        for head in 0..heads {
            let start = position * width + head * head_width;
            for component in 0..head_width {
                let rotated = if component < half {
                    -input[start + component + half]
                } else {
                    input[start + component - half]
                };
                let table = position * head_width + component;
                let cosine_product = mode.round(input[start + component] * cosine[table])?;
                let sine_product = mode.round(rotated * sine[table])?;
                output[start + component] = mode.round(cosine_product + sine_product)?;
            }
        }
    }
    Ok(output)
}

pub fn attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    positions: usize,
    heads: usize,
    head_width: usize,
    mode: Mode,
) -> io::Result<Vec<f32>> {
    let width = heads * head_width;
    if head_width == 0
        || query.len() != positions * width
        || key.len() != query.len()
        || value.len() != query.len()
    {
        return Err(io::Error::other("attention dimensions differ"));
    }
    let scaling = (head_width as f32).powf(-0.5);
    let mut output = vec![0.0; query.len()];
    let mut scores = vec![0.0; positions];
    let mut probability = vec![0.0; positions];
    for head in 0..heads {
        for query_position in 0..positions {
            let query_start = query_position * width + head * head_width;
            let mut maximum = f32::NEG_INFINITY;
            for (key_position, score) in scores.iter_mut().enumerate().take(query_position + 1) {
                let key_start = key_position * width + head * head_width;
                let mut dot = 0.0f32;
                for component in 0..head_width {
                    dot += query[query_start + component] * key[key_start + component];
                }
                let rounded_dot = mode.round(dot)?;
                let scaled = mode.round(rounded_dot * scaling)?;
                let masked = mode.round(scaled + 0.0)?;
                *score = masked;
                maximum = maximum.max(masked);
            }
            let mut denominator = 0.0f32;
            for key_position in 0..=query_position {
                probability[key_position] = (scores[key_position] - maximum).exp();
                denominator += probability[key_position];
            }
            for item in &mut probability[..=query_position] {
                *item = mode.round(*item / denominator)?;
            }
            for component in 0..head_width {
                let mut sum = 0.0f32;
                for (key_position, weight) in
                    probability.iter().enumerate().take(query_position + 1)
                {
                    let value_start = key_position * width + head * head_width;
                    sum += weight * value[value_start + component];
                }
                output[query_start + component] = mode.round(sum)?;
            }
        }
    }
    Ok(output)
}

pub fn residual(input: &[f32], branch: &[f32], mode: Mode) -> io::Result<Vec<f32>> {
    if input.len() != branch.len() {
        return Err(io::Error::other("residual dimensions differ"));
    }
    input
        .iter()
        .zip(branch)
        .map(|(value, addition)| mode.round(*value + *addition))
        .collect()
}

pub fn gated_silu(gate: &[f32], up_projection: &[f32], mode: Mode) -> io::Result<Vec<f32>> {
    if gate.len() != up_projection.len() {
        return Err(io::Error::other("feedforward dimensions differ"));
    }
    gate.iter()
        .zip(up_projection)
        .map(|(activation, multiplier)| {
            let silu = mode.round(*activation / (1.0 + (-*activation).exp()))?;
            mode.round(silu * *multiplier)
        })
        .collect()
}

pub fn contrast(reference: &[f32], terminal: &[f32], strength: f32) -> io::Result<Vec<f32>> {
    if reference.len() != terminal.len() || !strength.is_finite() {
        return Err(io::Error::other("contrast dimensions or strength differ"));
    }
    terminal
        .iter()
        .zip(reference)
        .map(|(terminal_value, reference_value)| {
            let difference = *terminal_value - *reference_value;
            let scaled = strength * difference;
            Mode::F32.round(*terminal_value + scaled)
        })
        .collect()
}
