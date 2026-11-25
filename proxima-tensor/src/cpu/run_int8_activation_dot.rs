use half::f16;
use crate::bind::Layout;
use proxima_numeric::block_layout::BlockLayout;
use proxima_numeric::codec::Codec;
use proxima_numeric::float_layout::FloatLayout;

use super::*;

const DOT_ELEMENTS: usize = 32;

/// Producers a packed-weight product can read as an already-quantized activation: every
/// `BlockQuantize` node, keyed by the node whose buffer holds its bytes (one f32 slot per byte).
pub(super) fn quantized_activation_layouts(ops: &[BoundOp]) -> BTreeMap<NodeId, BlockLayout> {
    ops.iter()
        .filter_map(|op| match &op.kind {
            BoundOpKind::BlockQuantize { layout, .. } if is_int8_activation(layout) => {
                Some((op.node, **layout))
            }
            _ => None,
        })
        .collect()
}

fn is_int8_activation(layout: &BlockLayout) -> bool {
    layout.element.code == FloatLayout::INT8
        && layout.scale.layout == FloatLayout::F16
        && layout.min.is_none()
        && layout.sub.is_none()
        && (0..layout.runs()).all(|run| layout.run_code_byte(run) >= layout.run_scale_byte(run) + 4)
}

/// Mirrors omega's `block_dot_i8` for the layouts whose codes are whole signed bytes under one
/// f16 scale: no min, no sub-blocks, no second plane, 32-element chunks.
fn weight_has_byte_code_dot(layout: &BlockLayout) -> bool {
    layout.element.code == FloatLayout::INT8
        && layout.scale.layout == FloatLayout::F16
        && layout.min.is_none()
        && layout.sub.is_none()
        && layout.scale_runs.is_none()
        && layout.element.bias == 0
        && layout.multiplier == 1
        && usize::from(layout.elements).is_multiple_of(DOT_ELEMENTS)
        && layout.contiguous_byte_codes().is_some()
}

fn half_at(bytes: &[u8], at: usize) -> Option<f32> {
    let pair = bytes.get(at..at + 2)?;
    Some(f16::from_le_bytes([pair[0], pair[1]]).to_f32())
}

/// One output element: sum over 32-element chunks of `(weight_scale * activation_scale) * sum(weight_code * activation_code)`,
/// the int32 chunk sum exact and the fold in f32, as omega's int8 gemm folds it. `None` when a row is short.
/// `chunk_sum` is the int32 chunk dot; every implementation returns the same exact integer.
fn int8_dot(
    weight: &BlockLayout,
    weight_row: &[u8],
    activation: &BlockLayout,
    activation_row: &[u8],
    depth: usize,
    chunk_sum: fn(&[u8], &[u8]) -> Option<i32>,
) -> Option<f32> {
    let weight_codes = weight.contiguous_byte_codes()?;
    let weight_elements = usize::from(weight.elements);
    let runs = usize::from(activation.runs());
    let mut accumulator = 0.0;
    for chunk in 0..depth / DOT_ELEMENTS {
        let element = chunk * DOT_ELEMENTS;
        let weight_block = weight_row.get(element / weight_elements * usize::from(weight.block_bytes)..)?;
        let weight_scale = half_at(weight_block, usize::from(weight.scale.byte_offset))?;
        let code_start = weight_codes.base_byte + element % weight_elements;
        let weight_chunk = weight_block.get(code_start..code_start + DOT_ELEMENTS)?;
        let run = u16::try_from(chunk % runs).ok()?;
        let activation_block = activation_row.get(chunk / runs * usize::from(activation.block_bytes)..)?;
        let activation_scale = half_at(activation_block, activation.run_scale_byte(run))?;
        let activation_start = activation.run_code_byte(run);
        let activation_chunk = activation_block.get(activation_start..activation_start + DOT_ELEMENTS)?;
        accumulator += (weight_scale * activation_scale) * chunk_sum(weight_chunk, activation_chunk)? as f32;
    }
    Some(accumulator)
}

/// The packed-product geometry both the production path and the test oracle walk.
struct Int8Product<'a> {
    weight: BlockLayout,
    activation: &'a BlockLayout,
    rows: usize,
    tokens: usize,
    depth: usize,
    weight_row_bytes: usize,
    activation_row_bytes: usize,
}

impl Int8Product<'_> {
    fn dot(
        &self,
        node: NodeId,
        weight_bytes: &[u8],
        activation_bytes: &[u8],
        (row, token): (usize, usize),
        chunk_sum: fn(&[u8], &[u8]) -> Option<i32>,
    ) -> Result<f32, TensorError> {
        let weight_row = &weight_bytes[row * self.weight_row_bytes..][..self.weight_row_bytes];
        let activation_row = &activation_bytes[token * self.activation_row_bytes..][..self.activation_row_bytes];
        int8_dot(&self.weight, weight_row, self.activation, activation_row, self.depth, chunk_sum).ok_or(
            TensorError::NotLowerable { node, reason: "int8 activation product ran past a packed row" },
        )
    }
}

fn int8_product<'a>(
    node: NodeId,
    weight_codec: Codec,
    weight_len: usize,
    activation: &'a BlockLayout,
    activation_len: usize,
    depth: usize,
    output_len: Option<usize>,
) -> Result<Int8Product<'a>, TensorError> {
    let weight = weight_codec
        .block_layout()
        .filter(weight_has_byte_code_dot)
        .ok_or(TensorError::ActivationQuantizedWeightCodec {
            node,
            codec: weight_codec.name().unwrap_or("unnamed block layout"),
        })?;
    let shape_error = TensorError::NotLowerable {
        node,
        reason: "int8 activation product rows do not tile the packed weight and activation bytes",
    };
    let weight_row_bytes = depth / usize::from(weight.elements) * usize::from(weight.block_bytes);
    let activation_row_bytes = depth / usize::from(activation.elements) * usize::from(activation.block_bytes);
    let aligned = depth.is_multiple_of(usize::from(activation.elements))
        && depth.is_multiple_of(usize::from(weight.elements));
    if !aligned || weight_row_bytes == 0 || activation_row_bytes == 0 {
        return Err(shape_error);
    }
    let rows = weight_len / weight_row_bytes;
    let tokens = activation_len / activation_row_bytes;
    if output_len.is_some_and(|len| len != rows * tokens) {
        return Err(shape_error);
    }
    Ok(Int8Product { weight, activation, rows, tokens, depth, weight_row_bytes, activation_row_bytes })
}

/// Workers the product splits over: the interpreter's shared matmul policy, so a product below
/// its parallel threshold, or with fewer weight rows than workers, stays on the calling thread.
fn product_workers(rows: usize, tokens: usize, depth: usize) -> Option<usize> {
    quantized_matmul_workers(rows, tokens.saturating_mul(depth))
}

/// `[tokens, rows]` product of packed weight rows against the rows of a `BlockQuantize` output,
/// computed as the int8 dot over the stored codes and scales. Refuses a weight codec without a
/// byte-code int8 dot rather than dequantizing it. Past the interpreter's parallel threshold the
/// weight rows split into `UNIT_ROWS` units over its worker pool, each unit owning a disjoint
/// `[row, token]` slot that is then scattered into the `[token, row]` output; every element is
/// the same chunk-ordered fold as the serial loop, so the split never changes a result.
pub(super) fn matmul_int8_activation(
    node: NodeId,
    weight_codec: Codec,
    weight_bytes: &[u8],
    activation: &BlockLayout,
    activation_bytes: &[u8],
    depth: usize,
    output: &mut [f32],
) -> Result<(), TensorError> {
    let product = int8_product(node, weight_codec, weight_bytes.len(), activation, activation_bytes.len(), depth, Some(output.len()))?;
    let (rows, tokens) = (product.rows, product.tokens);
    let Some(workers) = product_workers(rows, tokens, depth) else {
        for (token, token_output) in output.chunks_mut(rows).enumerate() {
            for (row, slot) in token_output.iter_mut().enumerate() {
                *slot = product.dot(node, weight_bytes, activation_bytes, (row, token), code_sum)?;
            }
        }
        return Ok(());
    };
    let by_unit = matmul_rows_threaded(rows.div_ceil(UNIT_ROWS), UNIT_ROWS * tokens, workers, None, depth, |unit, slot| {
        let first_row = unit * UNIT_ROWS;
        for (offset, row_slots) in slot.chunks_mut(tokens).take(UNIT_ROWS.min(rows - first_row)).enumerate() {
            for (token, value) in row_slots.iter_mut().enumerate() {
                *value = product.dot(node, weight_bytes, activation_bytes, (first_row + offset, token), code_sum)?;
            }
        }
        Ok(())
    })?;
    for (row, row_values) in by_unit.chunks(tokens).take(rows).enumerate() {
        for (token, value) in row_values.iter().enumerate() {
            output[token * rows + row] = *value;
        }
    }
    Ok(())
}

/// The pre-threading product: one scalar `token / row` loop, returning each element's bits in
/// `[token, row]` order. Test oracle for the threaded path.
#[cfg(test)]
fn matmul_int8_activation_serial(
    node: NodeId,
    weight_codec: Codec,
    weight_bytes: &[u8],
    activation: &BlockLayout,
    activation_bytes: &[u8],
    depth: usize,
) -> Result<Vec<u32>, TensorError> {
    let product = int8_product(node, weight_codec, weight_bytes.len(), activation, activation_bytes.len(), depth, None)?;
    let mut bits = Vec::new();
    for token in 0..product.tokens {
        for row in 0..product.rows {
            let value = product.dot(node, weight_bytes, activation_bytes, (row, token), |weight, activation| {
                Some(scalar_code_sum(weight, activation))
            })?;
            bits.push(value.to_bits());
        }
    }
    Ok(bits)
}

struct ProductShape {
    depth: usize,
    rows: usize,
    tokens: usize,
}

/// Row-major `[tokens, depth]` reads, in element units, over the reduction axes taken outer to
/// inner: each axis strides by the product of the extents nested inside it.
fn is_dense_run(read: &Layout, reduction_axes: &[u16], extents: &[u64]) -> bool {
    reduction_axes
        .iter()
        .rev()
        .try_fold(1i64, |expected, axis| {
            (read.stride(*axis) == expected).then(|| expected * extents[usize::from(*axis)] as i64)
        })
        .is_some()
}

/// The `[tokens, rows]` product this reduce computes, or `None` unless the bound addressing is
/// exactly what [`matmul_int8_activation`] assumes: two output axes ordered token then row, both
/// operands ungathered, the weight row-major `[rows, depth]` (token stride 0), the activation
/// row-major `[tokens, depth]` (row stride 0), and the output row-major `[tokens, rows]`.
fn dense_product_shape(resolved: &BoundOp, weight_node: NodeId, activation_node: NodeId) -> Option<ProductShape> {
    let BoundOpKind::Reduce { output_axes, out_layout, out_scatter: None, .. } = &resolved.kind else {
        return None;
    };
    let [token_axis, row_axis] = output_axes.as_slice() else {
        return None;
    };
    let extents = resolved.extents.as_slice();
    let read_of = |node: NodeId| resolved.operands().iter().find(|(candidate, _, _)| *candidate == node);
    let (_, weight_read, weight_gather) = read_of(weight_node)?;
    let (_, activation_read, activation_gather) = read_of(activation_node)?;
    if weight_gather.is_some() || activation_gather.is_some() {
        return None;
    }
    let reduction_axes: Vec<u16> = (0..u16::try_from(extents.len()).ok()?)
        .filter(|axis| !output_axes.contains(axis))
        .collect();
    let depth: u64 = reduction_axes.iter().map(|axis| extents[usize::from(*axis)]).product();
    let tokens = *extents.get(usize::from(*token_axis))?;
    let rows = *extents.get(usize::from(*row_axis))?;
    if depth == 0 || tokens == 0 || rows == 0 || token_axis == row_axis {
        return None;
    }
    let (depth_stride, rows_stride) = (i64::try_from(depth).ok()?, i64::try_from(rows).ok()?);
    let weight_ok = weight_read.base == 0
        && weight_read.stride(*token_axis) == 0
        && weight_read.stride(*row_axis) == depth_stride
        && is_dense_run(weight_read, &reduction_axes, extents);
    let activation_ok = activation_read.base == 0
        && activation_read.stride(*row_axis) == 0
        && activation_read.stride(*token_axis) == depth_stride
        && is_dense_run(activation_read, &reduction_axes, extents);
    let output_ok =
        out_layout.base == 0 && out_layout.stride(*token_axis) == rows_stride && out_layout.stride(*row_axis) == 1;
    (weight_ok && activation_ok && output_ok).then_some(ProductShape {
        depth: usize::try_from(depth).ok()?,
        rows: usize::try_from(rows).ok()?,
        tokens: usize::try_from(tokens).ok()?,
    })
}

fn packed_row_bytes(layout: &BlockLayout, depth: usize) -> usize {
    depth / usize::from(layout.elements) * usize::from(layout.block_bytes)
}

/// Runs `resolved` here when it is an ungathered packed-weight product whose activation operand
/// is a `BlockQuantize` output and whose addressing is the `[tokens, rows]` row-major product
/// [`dense_product_shape`] checks. `Ok(false)` declines (a gathered or expert-sourced weight, a
/// weight that is not dense packed bytes, any other axis order or stride, a scatter, a composed
/// body); the caller then runs [`refuse_quantized_activation_reader`], so a declined quantized
/// activation never reaches an interpreter that would read its bytes as f32.
pub(super) fn run_if_int8_activation<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    quantized_weights: &BTreeMap<NodeId, QuantizedBlock>,
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
    activation_layouts: &BTreeMap<NodeId, BlockLayout>,
    output: &mut [f32],
    numeric_policy: NumericPolicy,
) -> Result<bool, TensorError> {
    let Some(weight_node) = elementwise_matmul::quantized_operand(resolved, quantized_weights) else {
        return Ok(false);
    };
    let Ok(activation_node) = packed_reduce_activation_operand(resolved, weight_node) else {
        return Ok(false);
    };
    let Some(activation_layout) = activation_layouts.get(&activation_node) else {
        return Ok(false);
    };
    let Some(QuantizedBlock::Packed { codec, bytes }) = quantized_weights.get(&weight_node).copied() else {
        return Ok(false);
    };
    if expert_sources.is_some_and(|sources| sources.contains_key(&weight_node)) {
        return Ok(false);
    }
    let Some(shape) = dense_product_shape(resolved, weight_node, activation_node) else {
        return Ok(false);
    };
    let packed: Vec<u8> = buffer_of(buffers, activation_node)?
        .iter()
        .map(|slot| *slot as u8)
        .collect();
    let weight_sized = codec
        .block_layout()
        .is_none_or(|layout| bytes.len() == shape.rows * packed_row_bytes(&layout, shape.depth));
    if !weight_sized || packed.len() != shape.tokens * packed_row_bytes(activation_layout, shape.depth) {
        return Ok(false);
    }
    matmul_int8_activation(resolved.node, codec, bytes, activation_layout, &packed, shape.depth, output)?;
    apply_reduce_epilogue(resolved, buffers, output, numeric_policy)?;
    Ok(true)
}

/// Refuses `resolved` when it reads a `BlockQuantize` int8 activation, for every interpreter that
/// has no int8 dot (the arena, the parallel evaluator, and the packed path's own declines).
pub(super) fn refuse_quantized_activation_reader(
    resolved: &BoundOp,
    activation_layouts: &BTreeMap<NodeId, BlockLayout>,
) -> Result<(), TensorError> {
    match resolved
        .all_read_sources()
        .find(|(node, _, _)| activation_layouts.contains_key(node))
    {
        Some((activation, _, _)) => Err(TensorError::QuantizedActivationUnserved {
            node: resolved.node,
            activation: *activation,
        }),
        None => Ok(()),
    }
}

/// [`refuse_quantized_activation_reader`] over every node of `resolved` that is not in `dead`.
/// The static arena runs this once at build, so no arena path (the static-node pass, the f64
/// island, a fused epilogue consumer, the masked probe) can evaluate a reader of an int8 activation.
pub(super) fn refuse_live_quantized_activation_readers(
    resolved: &[BoundOp],
    dead: &BTreeSet<NodeId>,
) -> Result<(), TensorError> {
    let activation_layouts = quantized_activation_layouts(resolved);
    resolved
        .iter()
        .filter(|computed| !dead.contains(&computed.node))
        .try_for_each(|computed| refuse_quantized_activation_reader(computed, &activation_layouts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BodyStep;
    use proxima_numeric::ScaleArithmetic;
    use crate::test_support::Lcg;
    use proxima_gguf::quant::block_layout_encode;

    const NODE: NodeId = NodeId(7);

    /// Worked example, one real-shaped 32-element activation chunk against one Q8_0 weight chunk.
    ///
    /// Activation x: codes c = [127,-100,64,-64,32,-32,16,-16,8,-8,4,-4,2,-2,1,-1, 0 x8,
    /// 50,-50,75,-75,90,-90,110,-110], x[i] = c[i] / 100, except even i > 0 carry +0.003
    /// (0.3 of a code, so rounding is unambiguous) and x[0] = 1.27 is the peak.
    ///   amax = 1.27, d = amax / 127 = 0.01, q = round(x / d) = c (half away from zero).
    ///   stored scale = f16(0.01) = 1311 * 2^-17 = 0.010002136 (0.01 / 2^-17 = 1310.72 -> 1311).
    ///   sum q = 127 - 100 = 27 (every other pair cancels), stored sum = f16(d * 27) = f16(0.27)
    ///   = 1106 * 2^-12 = 0.27001953.
    /// Weight Q8_0: scale 0.03125 (exactly 2^-5), codes qw[i] = [3,-2,1,-1][i % 4].
    ///   sum qw * q per 8-element run of the period: 0..7 -> 381+200+64+64+96+64+16+16 = 901,
    ///   8..15 -> 24+16+4+4+6+4+1+1 = 60, 16..23 -> 0, 24..31 -> 150+100+75+75+270+180+110+110 = 1070,
    ///   total 901 + 60 + 0 + 1070 = 2031 (running sum after element 31 is 2031).
    /// Output = (0.03125 * 0.010002136) * 2031 = (1311 / 4194304) * 2031 = 2662641 / 4194304
    /// = 0.63482255..., every factor dyadic and 2662641 < 2^24, so the f32 value is exact.
    fn worked_example() -> (Vec<f32>, Vec<u8>, Vec<u8>) {
        let codes: [i8; 32] = [
            127, -100, 64, -64, 32, -32, 16, -16, 8, -8, 4, -4, 2, -2, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 50,
            -50, 75, -75, 90, -90, 110, -110,
        ];
        let activation: Vec<f32> = codes
            .iter()
            .enumerate()
            .map(|(index, code)| {
                let nudge = if index > 0 && index % 2 == 0 { 0.003 } else { 0.0 };
                if index == 0 { 1.27 } else { f32::from(*code) / 100.0 + nudge }
            })
            .collect();
        let mut block = vec![0u8; 36];
        block[0..2].copy_from_slice(&f16::from_f32(0.01).to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(0.27).to_le_bytes());
        block[4..].iter_mut().zip(codes).for_each(|(slot, code)| *slot = code as u8);
        let weight_pattern = [3i8, -2, 1, -1];
        let mut weight = vec![0u8; 34];
        weight[0..2].copy_from_slice(&f16::from_f32(0.03125).to_le_bytes());
        weight[2..].iter_mut().enumerate().for_each(|(index, slot)| *slot = weight_pattern[index % 4] as u8);
        (activation, block, weight)
    }

    #[test]
    fn worked_example_q8_0_row_against_q8_1_chunk_is_exactly_2662641_over_2_pow_22() {
        let (activation, block, weight) = worked_example();
        let mut encoded = vec![0u8; 36];
        block_layout_encode::quantize(&BlockLayout::Q8_1, &activation, &mut encoded).expect("q8_1 encodes");
        assert_eq!(encoded, block);
        let mut output = [0.0f32];
        matmul_int8_activation(NODE, q8_0(), &weight, &BlockLayout::Q8_1, &block, 32, &mut output)
            .expect("q8_0 has a byte-code int8 dot");
        assert_eq!(output[0], 2_662_641.0 / 4_194_304.0);
    }

    fn q8_0_rows(random: &mut Lcg, rows: usize, depth: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        for _ in 0..rows * depth / 32 {
            bytes.extend_from_slice(&f16::from_f32(0.002 + 0.01 * random.next_unit().abs()).to_le_bytes());
            bytes.extend((0..32).map(|_| (random.next_unit() * 127.0) as i8 as u8));
        }
        bytes
    }

    fn activations(random: &mut Lcg, tokens: usize, depth: usize) -> Vec<f32> {
        (0..tokens * depth)
            .map(|index| if (index / 32) % 5 == 3 { 0.0 } else { random.next_unit() * (1.0 + (index % 7) as f32) })
            .collect()
    }

    fn quantize(layout: &BlockLayout, values: &[f32], tokens: usize, depth: usize) -> Vec<u8> {
        let row_bytes = depth / usize::from(layout.elements) * usize::from(layout.block_bytes);
        let mut packed = vec![0u8; tokens * row_bytes];
        block_layout_encode::quantize(layout, values, &mut packed).expect("finite activations encode");
        packed
    }

    fn oracle(weight: &[u8], values: &[f32], rows: usize, tokens: usize, depth: usize) -> (Vec<f64>, Vec<f64>) {
        let mut results = Vec::new();
        let mut magnitudes = Vec::new();
        for token in 0..tokens {
            let mut quantized = Vec::new();
            for chunk in values[token * depth..][..depth].chunks(32) {
                let amax = chunk.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
                let step = amax / 127.0;
                let stored = f16::from_f32(step).to_f32();
                quantized.extend(chunk.iter().map(|value| {
                    if amax == 0.0 { 0.0 } else { f64::from((value / step).round()) * f64::from(stored) }
                }));
            }
            for row in 0..rows {
                let (mut total, mut magnitude) = (0.0f64, 0.0f64);
                for (chunk, block) in weight[row * depth / 32 * 34..][..depth / 32 * 34].chunks(34).enumerate() {
                    let scale = f64::from(f16::from_le_bytes([block[0], block[1]]).to_f32());
                    for (offset, code) in block[2..].iter().enumerate() {
                        let term = scale * f64::from(*code as i8) * quantized[chunk * 32 + offset];
                        total += term;
                        magnitude += term.abs();
                    }
                }
                results.push(total);
                magnitudes.push(magnitude);
            }
        }
        (results, magnitudes)
    }

    #[test]
    fn q8_0_matvec_over_q8_1_activation_matches_the_fake_quantize_oracle_within_f32_accumulation() {
        let (rows, tokens, depth) = (6, 3, 256);
        let mut random = Lcg(0x389);
        let weight = q8_0_rows(&mut random, rows, depth);
        let values = activations(&mut random, tokens, depth);
        let packed = quantize(&BlockLayout::Q8_1, &values, tokens, depth);
        let mut output = vec![0.0f32; rows * tokens];
        matmul_int8_activation(NODE, q8_0(), &weight, &BlockLayout::Q8_1, &packed, depth, &mut output)
            .expect("q8_0 dots against q8_1");
        let (expected, magnitudes) = oracle(&weight, &values, rows, tokens, depth);
        for ((got, want), magnitude) in output.iter().zip(&expected).zip(&magnitudes) {
            let tolerance = magnitude * (depth / 32) as f64 * f64::from(f32::EPSILON) + 1e-12;
            assert!((f64::from(*got) - want).abs() <= tolerance, "{got} vs {want} +- {tolerance}");
        }
    }

    /// Four consecutive `Q8_1` blocks rewritten as one `Q8_1_GROUPED` block: a 16-byte header of
    /// four (scale, sum) pairs, then the four 32-code runs. Built from the flat bytes so the dot's
    /// run addressing is checked on identical stored scales and codes; the grouped encoder rounds
    /// near-ties differently from the flat one, so it cannot supply the grouped bytes here.
    fn regroup(single: &[u8]) -> Vec<u8> {
        single
            .chunks(4 * 36)
            .flat_map(|group| {
                let header = group.chunks(36).flat_map(|block| block[..4].iter().copied());
                let codes = group.chunks(36).flat_map(|block| block[4..].iter().copied());
                header.chain(codes).collect::<Vec<u8>>()
            })
            .collect()
    }

    #[test]
    fn grouped_q8_1_activation_equals_four_q8_1_runs() {
        let (rows, tokens, depth) = (4, 2, 512);
        let mut random = Lcg(0x1a2b);
        let weight = q8_0_rows(&mut random, rows, depth);
        let values = activations(&mut random, tokens, depth);
        let mut flat = vec![0.0f32; rows * tokens];
        let mut grouped = vec![0.0f32; rows * tokens];
        let single = quantize(&BlockLayout::Q8_1, &values, tokens, depth);
        let together = regroup(&single);
        matmul_int8_activation(NODE, q8_0(), &weight, &BlockLayout::Q8_1, &single, depth, &mut flat).expect("flat");
        matmul_int8_activation(NODE, q8_0(), &weight, &BlockLayout::Q8_1_GROUPED, &together, depth, &mut grouped)
            .expect("grouped");
        assert_eq!(flat, grouped);
    }

    const TOKENS: usize = 2;
    const ROWS: usize = 3;
    const DEPTH: usize = 64;

    struct Product {
        weight: Vec<u8>,
        activation: Vec<f32>,
        packed: Vec<u8>,
    }

    fn product() -> Product {
        let mut random = Lcg(0x390);
        let weight = q8_0_rows(&mut random, ROWS, DEPTH);
        let values = activations(&mut random, TOKENS, DEPTH);
        let packed = quantize(&BlockLayout::Q8_1, &values, TOKENS, DEPTH);
        let activation = packed.iter().map(|byte| f32::from(*byte)).collect();
        Product { weight, activation, packed }
    }

    fn read(strides: [i64; 3]) -> Layout {
        Layout { base: 0, strides: strides.into_iter().collect() }
    }

    fn lookup() -> bind::Lookup {
        bind::Lookup { indices: NodeId(9), index_layout: read([1, 0, 0]), element_stride: 1, extent: 3 }
    }

    struct Wiring {
        output_axes: [u16; 2],
        out_strides: [i64; 3],
        weight_strides: [i64; 3],
        weight_gather: Option<bind::Lookup>,
    }

    fn token_row_wiring() -> Wiring {
        Wiring {
            output_axes: [0, 1],
            out_strides: [ROWS as i64, 1, 0],
            weight_strides: [0, DEPTH as i64, 1],
            weight_gather: None,
        }
    }

    fn reduce(wiring: &Wiring) -> BoundOp {
        BoundOp {
            node: NodeId(2),
            dtype: DType::FLOAT32,
            extents: vec![TOKENS as u64, ROWS as u64, DEPTH as u64],
            kind: BoundOpKind::Reduce {
                element_body: ComposedBody {
                    steps: vec![BodyStep { op: ScalarOp::Multiply, args: vec![StepArg::Operand(0), StepArg::Operand(1)] }],
                },
                reduce_op: ScalarOp::Add,
                init: ReduceInit::Zero,
                keep: Keep::Reduce,
                operands: vec![
                    (NodeId(0), read(wiring.weight_strides), wiring.weight_gather.clone()),
                    (NodeId(1), read([DEPTH as i64, 0, 1]), None),
                ],
                output_axes: wiring.output_axes.into_iter().collect(),
                out_layout: read(wiring.out_strides),
                out_scatter: None,
                epilogue_body: ComposedBody::leaf(ScalarOp::Identity),
                epilogue_operands: Vec::new(),
                epilogue_broadcast_axes: smallvec::smallvec![],
                accumulator: DType::FLOAT32,
            },
        }
    }

    fn run(wiring: &Wiring) -> (Result<bool, TensorError>, Vec<f32>, Result<(), TensorError>) {
        let data = product();
        let resolved = reduce(wiring);
        let buffers: Vec<Option<Cow<'_, [f32]>>> =
            vec![None, Some(Cow::Borrowed(data.activation.as_slice())), None];
        let weights = BTreeMap::from([(NodeId(0), QuantizedBlock::Packed { codec: q8_0(), bytes: &data.weight })]);
        let layouts = BTreeMap::from([(NodeId(1), BlockLayout::Q8_1)]);
        let mut output = vec![0.0f32; TOKENS * ROWS];
        let handled = run_if_int8_activation(&resolved, &buffers, &weights, None, &layouts, &mut output, UNPLANNED_KERNEL_POLICY);
        (handled, output, refuse_quantized_activation_reader(&resolved, &layouts))
    }

    #[test]
    fn int8_activation_reduce_in_token_row_order_runs_as_the_int8_dot() {
        let data = product();
        let mut expected = vec![0.0f32; TOKENS * ROWS];
        matmul_int8_activation(NODE, q8_0(), &data.weight, &BlockLayout::Q8_1, &data.packed, DEPTH, &mut expected)
            .expect("q8_0 dots against q8_1");
        let (handled, output, _) = run(&token_row_wiring());
        assert!(handled.expect("dense product runs"));
        assert_eq!(output, expected);
    }

    #[test]
    fn int8_activation_reduce_with_a_gathered_weight_declines_and_is_refused() {
        let wiring = Wiring { weight_gather: Some(lookup()), ..token_row_wiring() };
        let (handled, output, refusal) = run(&wiring);
        assert!(!handled.expect("a gathered weight declines"));
        assert!(output.iter().all(|value| *value == 0.0), "a declined product writes nothing");
        assert!(matches!(refusal, Err(TensorError::QuantizedActivationUnserved { activation: NodeId(1), .. })));
    }

    #[test]
    fn int8_activation_reduce_in_row_token_order_declines_and_is_refused() {
        let wiring = Wiring { output_axes: [1, 0], out_strides: [1, TOKENS as i64, 0], ..token_row_wiring() };
        let (handled, _, refusal) = run(&wiring);
        assert!(!handled.expect("a row-major output declines"));
        assert!(matches!(refusal, Err(TensorError::QuantizedActivationUnserved { .. })));
    }

    #[test]
    fn int8_activation_reduce_over_a_padded_weight_row_stride_declines() {
        let wiring = Wiring { weight_strides: [0, DEPTH as i64 + 32, 1], ..token_row_wiring() };
        let (handled, _, _) = run(&wiring);
        assert!(!handled.expect("a padded weight stride declines"));
    }

    #[test]
    fn int8_activation_reduce_over_a_transposed_output_stride_declines() {
        let wiring = Wiring { out_strides: [1, TOKENS as i64, 0], ..token_row_wiring() };
        let (handled, _, _) = run(&wiring);
        assert!(!handled.expect("a transposed output stride declines"));
    }

    #[test]
    fn a_reduce_that_reads_no_quantized_activation_is_not_refused() {
        let resolved = reduce(&token_row_wiring());
        assert!(refuse_quantized_activation_reader(&resolved, &BTreeMap::new()).is_ok());
    }

    fn quantizer() -> BoundOp {
        let bytes = (DEPTH / 32 * 36) as u64;
        BoundOp {
            node: NodeId(1),
            dtype: DType::UINT8,
            extents: vec![TOKENS as u64, bytes],
            kind: BoundOpKind::BlockQuantize {
                operands: vec![(NodeId(3), Layout { base: 0, strides: [DEPTH as i64, 1].into_iter().collect() }, None)],
                layout: Box::new(BlockLayout::Q8_1),
                arithmetic: ScaleArithmetic::Quotient,
            },
        }
    }

    fn interpreter_buffers() -> Vec<Option<Vec<f32>>> {
        let mut random = Lcg(0x391);
        vec![None, None, None, Some(activations(&mut random, TOKENS, DEPTH))]
    }

    fn unserved(result: Result<(), TensorError>) -> bool {
        matches!(
            result,
            Err(TensorError::QuantizedActivationUnserved { node: NodeId(2), activation: NodeId(1) })
        )
    }

    #[test]
    fn static_arena_build_refuses_a_live_reader_of_a_quantized_activation() {
        let resolved = [quantizer(), reduce(&token_row_wiring())];
        let refusal = refuse_live_quantized_activation_readers(&resolved, &BTreeSet::new());
        assert!(unserved(refusal));
    }

    #[test]
    fn static_arena_build_ignores_a_dead_reader_of_a_quantized_activation() {
        let resolved = [quantizer(), reduce(&token_row_wiring())];
        let dead = BTreeSet::from([NodeId(2)]);
        assert!(refuse_live_quantized_activation_readers(&resolved, &dead).is_ok());
    }

    #[test]
    fn interpreter_refuses_a_reader_folded_in_the_same_batch_as_its_quantizer() {
        let mut buffers = interpreter_buffers();
        let interpreter = Interpreter::new(&mut buffers);
        let batch: bind::ReadyBatch = [quantizer(), reduce(&token_row_wiring())].into_iter().collect();
        assert!(unserved(block_on(interpreter.call(batch))));
    }

    #[test]
    fn interpreter_refuses_a_reader_folded_in_a_later_batch_than_its_quantizer() {
        let mut buffers = interpreter_buffers();
        let interpreter = Interpreter::new(&mut buffers);
        let first: bind::ReadyBatch = [quantizer()].into_iter().collect();
        block_on(interpreter.call(first)).expect("the quantizer itself runs on the interpreter");
        let second: bind::ReadyBatch = [reduce(&token_row_wiring())].into_iter().collect();
        assert!(unserved(block_on(interpreter.call(second))));
    }

    #[test]
    fn interpreter_refuses_an_f64_island_reader_before_the_island_reads_the_bytes() {
        let mut buffers = interpreter_buffers();
        let interpreter = Interpreter::new(&mut buffers);
        let wide_reader = BoundOp { dtype: DType::FLOAT64, ..reduce(&token_row_wiring()) };
        let batch: bind::ReadyBatch = [quantizer(), wide_reader].into_iter().collect();
        assert!(unserved(block_on(interpreter.call(batch))));
    }

    #[test]
    fn a_weight_codec_without_a_byte_code_int8_dot_is_a_typed_refusal() {
        let mut output = [0.0f32; 1];
        let block = [0u8; 36];
        let error = matmul_int8_activation(NODE, Codec::Q4_0, &[0u8; 18], &BlockLayout::Q8_1, &block, 32, &mut output)
            .expect_err("q4_0 packs two codes per byte");
        assert!(matches!(
            error,
            TensorError::ActivationQuantizedWeightCodec { codec: "q4_0", .. }
        ));
    }

    macro_rules! bits {
        ($values:expr) => {
            $values.iter().map(|value| value.to_bits()).collect::<Vec<u32>>()
        };
    }

    fn q8_0() -> Codec {
        Codec::Q8_0
    }

    fn both_paths(weight: &[u8], packed: &[u8], rows: usize, tokens: usize, depth: usize) -> (Vec<u32>, Vec<u32>) {
        let mut new = vec![0.0; rows * tokens];
        let old = matmul_int8_activation_serial(NODE, q8_0(), weight, &BlockLayout::Q8_1, packed, depth).expect("serial oracle");
        matmul_int8_activation(NODE, q8_0(), weight, &BlockLayout::Q8_1, packed, depth, &mut new).expect("product");
        (old, bits!(new))
    }

    #[test]
    fn threaded_product_equals_the_serial_scalar_oracle_bit_for_bit_across_the_parallel_threshold() {
        let shapes = [(3, 1, 32), (5, 3, 64), (17, 1, 128), (40, 5, 256), (64, 7, 512), (130, 9, 1024), (257, 3, 2048), (33, 1, 4096)];
        let mut above = 0;
        for (index, (rows, tokens, depth)) in shapes.into_iter().enumerate() {
            let mut random = Lcg(0x696 + index as u64);
            let weight = q8_0_rows(&mut random, rows, depth);
            let values = activations(&mut random, tokens, depth);
            let packed = quantize(&BlockLayout::Q8_1, &values, tokens, depth);
            let (old, new) = both_paths(&weight, &packed, rows, tokens, depth);
            assert_eq!(old, new, "rows {rows} tokens {tokens} depth {depth}");
            above += usize::from(rows * tokens * depth >= PARALLEL_THRESHOLD);
        }
        assert!(above >= 4, "only {above} shapes reached the threaded split");
    }

    #[test]
    fn full_range_codes_including_the_most_negative_byte_keep_the_simd_sum_exact() {
        let (rows, tokens, depth) = (48, 5, 512);
        let mut random = Lcg(0x6961);
        let layout = BlockLayout::Q8_1;
        let mut weight = q8_0_rows(&mut random, rows, depth);
        let weight_layout = q8_0().block_layout().expect("q8_0 is a block layout");
        for block in weight.chunks_mut(usize::from(weight_layout.block_bytes)) {
            block[weight_layout.contiguous_byte_codes().expect("byte codes").base_byte..].iter_mut().for_each(|code| {
                *code = (random.next_unit() * 128.0) as i8 as u8;
            });
        }
        let values = activations(&mut random, tokens, depth);
        let mut packed = quantize(&layout, &values, tokens, depth);
        for block in packed.chunks_mut(usize::from(layout.block_bytes)) {
            block[layout.run_code_byte(0)..].iter_mut().enumerate().for_each(|(slot, code)| {
                *code = if slot % 3 == 0 { 0x80 } else { (random.next_unit() * 128.0) as i8 as u8 };
            });
        }
        let (old, new) = both_paths(&weight, &packed, rows, tokens, depth);
        assert_eq!(old, new);
    }

    #[test]
    fn products_past_the_threshold_split_over_the_shared_worker_pool() {
        let configured = std::env::var_os("PROXIMA_MATMUL_WORKERS").is_some();
        let parallel = std::thread::available_parallelism().map_or(1, |count| count.get()) > 1;
        let (rows, tokens, depth) = (1024, 64, 1024);
        let workers = product_workers(rows, tokens, depth);
        assert_eq!(workers.is_some(), matmul_worker_count() > 1);
        if parallel && !configured {
            assert!(workers.is_some_and(|count| count > 1), "workers {workers:?}");
        }
        assert_eq!(product_workers(1, 1, 32), None);
        assert_eq!(product_workers(2, 1, PARALLEL_THRESHOLD / 4), None);
    }

    #[test]
    #[ignore = "micro-measure, run with --ignored --nocapture"]
    fn micro_measure_threaded_against_serial_scalar_product() {
        let (rows, tokens, depth) = (1024, 512, 1024);
        let mut random = Lcg(0x6962);
        let weight = q8_0_rows(&mut random, rows, depth);
        let values = activations(&mut random, tokens, depth);
        let packed = quantize(&BlockLayout::Q8_1, &values, tokens, depth);
        let mut new = vec![0.0; rows * tokens];
        let started = std::time::Instant::now();
        let old = matmul_int8_activation_serial(NODE, q8_0(), &weight, &BlockLayout::Q8_1, &packed, depth).expect("old");
        let old_wall = started.elapsed();
        let started = std::time::Instant::now();
        matmul_int8_activation(NODE, q8_0(), &weight, &BlockLayout::Q8_1, &packed, depth, &mut new).expect("new");
        let new_wall = started.elapsed();
        println!(
            "int8 product tokens {tokens} depth {depth} rows {rows}: old serial scalar {old_wall:?}, new {new_wall:?}, workers {:?}",
            product_workers(rows, tokens, depth)
        );
        assert_eq!(old, bits!(new));
    }
}
