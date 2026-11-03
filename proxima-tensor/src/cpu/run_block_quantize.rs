use proxima_gguf::quant::block_layout_encode;

use super::*;

pub(super) fn run_block_quantize<B: Deref<Target = [f32]> + Sync>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let BoundOpKind::BlockQuantize { operands, layout } = &resolved.kind else {
        return Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "block quantize runner received another bound operation",
        });
    };
    let (source_node, _, _) = operands.first().ok_or(TensorError::Empty)?;
    let source = buffer_of(buffers, *source_node)?;
    let mut packed = vec![0u8; output.len()];
    block_layout_encode::quantize(layout, source, &mut packed).map_err(|cause| {
        TensorError::BlockQuantizeLayout {
            node: resolved.node,
            cause,
        }
    })?;
    for (slot, byte) in output.iter_mut().zip(&packed) {
        *slot = f32::from(*byte);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bind::Layout;
    use crate::op::NodeId;
    use smallvec::SmallVec;
    use crate::test_support::Lcg;
    use proxima_gguf::quant::QuantError;
    use proxima_gguf::quant::block_layout::dequantize;
    use proxima_primitives::Codec;
    use proxima_primitives::block_layout::BlockLayout;

    const ROWS: u32 = 3;
    const BLOCKS_PER_ROW: u32 = 4;
    const F16_EPSILON: f32 = 1.0 / 1024.0;

    fn bound(layout: BlockLayout, last: u32) -> BoundOp {
        let stride = SmallVec::from_vec(vec![i64::from(last), 1]);
        let bytes = u64::from(last / u32::from(layout.elements)) * u64::from(layout.block_bytes);
        BoundOp {
            node: NodeId(1),
            dtype: DType::UINT8,
            extents: vec![u64::from(ROWS), bytes],
            kind: BoundOpKind::BlockQuantize {
                operands: vec![(NodeId(0), Layout { base: 0, strides: stride }, None)],
                layout: Box::new(layout),
            },
        }
    }

    fn row_width(layout: &BlockLayout) -> u32 {
        u32::from(layout.elements) * BLOCKS_PER_ROW
    }

    fn inputs(layout: &BlockLayout, seed: u64) -> Vec<f32> {
        let mut random = Lcg(seed);
        let count = (ROWS * row_width(layout)) as usize;
        (0..count)
            .map(|index| random.next_unit() * (1.0 + (index % 7) as f32))
            .collect()
    }

    fn encode(layout: &BlockLayout, values: &[f32]) -> Result<Vec<u8>, TensorError> {
        let op = bound(*layout, row_width(layout));
        let mut output = vec![0.0f32; op.extents.iter().product::<u64>() as usize];
        let buffers = [Some(values), None];
        run_block_quantize(&op, &buffers, &mut output)?;
        Ok(output.iter().map(|slot| *slot as u8).collect())
    }

    fn decoded(layout: &BlockLayout, packed: &[u8]) -> Vec<f32> {
        let blocks = packed.len() / usize::from(layout.block_bytes);
        let mut values = vec![0.0f32; blocks * usize::from(layout.elements)];
        dequantize(layout, packed, &mut values).expect("the packed run is whole blocks");
        values
    }

    fn linear_layouts() -> Vec<(Codec, BlockLayout)> {
        (0..=u8::MAX)
            .filter_map(Codec::from_tag)
            .filter_map(|codec| codec.block_layout().map(|layout| (codec, layout)))
            .filter(|(_, layout)| block_layout_encode::reach(layout).is_ok())
            .collect()
    }

    fn half_at(block: &[u8], at: usize) -> f32 {
        half::f16::from_le_bytes([block[at], block[at + 1]]).to_f32()
    }

    fn stated_error(layout: &BlockLayout, peak: f32) -> f32 {
        let reach = block_layout_encode::reach(layout).expect("a linear layout has a reach");
        f32::from(layout.multiplier) * (peak / reach) / 2.0 + 2.0 * peak * F16_EPSILON
    }

    #[test]
    fn q8_1_block_matches_a_hand_computation() {
        let layout = BlockLayout::Q8_1;
        let mut values = vec![0.0f32; (ROWS * row_width(&layout)) as usize];
        for (index, value) in values.iter_mut().take(32).enumerate() {
            *value = index as f32 - 16.0;
        }
        let packed = encode(&layout, &values).expect("q8_1 encodes");
        let scale = half::f16::from_f32(16.0 / 127.0).to_f32();
        let codes: Vec<i32> = (0..32)
            .map(|index| ((index as f32 - 16.0) / scale).round() as i32)
            .collect();
        assert_eq!(half_at(&packed, 0), scale);
        for (index, code) in codes.iter().enumerate() {
            assert_eq!(i32::from(packed[4 + index] as i8), *code);
        }
        let expected_sum = half::f16::from_f32(scale * codes.iter().sum::<i32>() as f32).to_f32();
        assert_eq!(half_at(&packed, 2), expected_sum);
        assert!(
            packed[36..72].iter().all(|byte| *byte == 0),
            "an all-zero block is all zero bytes"
        );
    }

    #[test]
    fn every_linear_codec_layout_round_trips_within_its_stated_error() {
        let layouts = linear_layouts();
        let codecs: Vec<Codec> = layouts.iter().map(|(codec, _)| *codec).collect();
        let expected = [
            Codec::Q8_0,
            Codec::Q4_0,
            Codec::Q5_0,
            Codec::Q8_1,
            Codec::Q8K,
            Codec::Q1_0,
            Codec::Q2_0,
            Codec::Pq2_0,
        ];
        assert_eq!(
            codecs, expected,
            "the codecs a shared absolute-maximum scale encodes"
        );
        for (codec, layout) in layouts {
            let values = inputs(&layout, 0x5eed ^ u64::from(layout.block_bytes));
            let restored = decoded(
                &layout,
                &encode(&layout, &values).expect("a linear layout encodes"),
            );
            let blocks = values
                .chunks(usize::from(layout.elements))
                .zip(restored.chunks(usize::from(layout.elements)));
            for (block, (originals, decodes)) in blocks.enumerate() {
                let peak = originals
                    .iter()
                    .fold(0.0f32, |peak, value| peak.max(value.abs()));
                let bound = stated_error(&layout, peak);
                for (original, decode) in originals.iter().zip(decodes) {
                    assert!(
                        (original - decode).abs() <= bound,
                        "{codec:?} block {block}: {original} decoded to {decode}, bound {bound}"
                    );
                }
            }
        }
    }

    #[test]
    fn encoding_a_decoded_run_reproduces_the_same_bytes() {
        for (codec, layout) in linear_layouts() {
            let values = inputs(&layout, 0xfeed ^ u64::from(layout.block_bytes));
            let first = encode(&layout, &values).expect("a linear layout encodes");
            let second =
                encode(&layout, &decoded(&layout, &first)).expect("the decoded run encodes");
            assert_eq!(first, second, "{codec:?} is not idempotent on its own grid");
        }
    }

    #[test]
    fn layouts_with_sub_blocks_or_a_min_are_a_typed_error() {
        let refused = [
            BlockLayout::Q4_1,
            BlockLayout::Q4_K,
            BlockLayout::Q6_K,
            BlockLayout::TQ1_0,
        ];
        for layout in refused {
            let error =
                encode(&layout, &inputs(&layout, 7)).expect_err("not a shared-scale linear layout");
            assert!(
                matches!(
                    error,
                    TensorError::BlockQuantizeLayout {
                        cause: QuantError::UnsupportedBlockLayout { .. },
                        ..
                    }
                ),
                "{error:?}"
            );
        }
    }

    #[test]
    fn a_non_finite_source_is_a_typed_error() {
        let layout = BlockLayout::Q8_0;
        let mut values = inputs(&layout, 3);
        values[5] = f32::NAN;
        let error = encode(&layout, &values).expect_err("nan has no code");
        assert!(matches!(
            error,
            TensorError::BlockQuantizeLayout {
                cause: QuantError::NonFiniteInput { .. },
                ..
            }
        ));
    }
}
