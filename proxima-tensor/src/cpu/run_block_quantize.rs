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
    use proxima_numeric::codec::Codec;
    use proxima_numeric::block_layout::BlockLayout;

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
        let op = bound(*layout, (values.len() / ROWS as usize) as u32);
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

    fn encodable_layouts() -> Vec<(Codec, BlockLayout)> {
        (0..=u8::MAX)
            .filter_map(Codec::from_tag)
            .filter_map(|codec| codec.block_layout().map(|layout| (codec, layout)))
            .filter(|(_, layout)| block_layout_encode::reach(layout).is_ok())
            .collect()
    }

    fn half_at(block: &[u8], at: usize) -> f32 {
        half::f16::from_le_bytes([block[at], block[at + 1]]).to_f32()
    }

    /// Narrow integer codes: one step of the ratio the layout's multiplier describes.
    fn integer_code(layout: &BlockLayout) -> bool {
        layout.element.code.exponent_bits == 0
    }

    /// The scale the encoder will store for this block peak, read back through the scale layout.
    fn scale_of(layout: &BlockLayout, peak: f32) -> f32 {
        let reach = block_layout_encode::reach(layout).expect("an encodable layout has a reach");
        let code = proxima_numeric::float_layout_codec::encode(peak / reach, layout.scale.layout);
        proxima_numeric::float_layout_codec::decode(code, layout.scale.layout)
    }

    fn stated_error(layout: &BlockLayout, peak: f32) -> f32 {
        if integer_code(layout) {
            let reach = block_layout_encode::reach(layout).expect("an encodable layout has a reach");
            f32::from(layout.multiplier) * (peak / reach) / 2.0 + 2.0 * peak * F16_EPSILON
        } else {
            // a float element code sits on a geometric grid: the widest step is at the top
            // exponent (two units for E2M1, four for E3M2), so half of it is two units of scale,
            // plus the scale's own rounding
            2.5 * scale_of(layout, peak) + 2.0 * peak * F16_EPSILON
        }
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
    fn grouped_q8_1_holds_four_independent_q8_1_runs_behind_one_aligned_header() {
        let grouped = BlockLayout::Q8_1_GROUPED;
        let values = inputs(&BlockLayout::Q8_1, 0xa11);
        let packed = encode(&grouped, &values).expect("the grouped layout encodes");
        let runs = encode(&BlockLayout::Q8_1, &values).expect("q8_1 encodes the same values");
        assert_eq!(packed.len(), values.len() / 128 * 144);
        for (group, block) in packed.chunks(144).enumerate() {
            for run in 0..4 {
                let single = &runs[(group * 4 + run) * 36..][..36];
                assert_eq!(&block[run * 4..][..4], &single[..4], "group {group} run {run} header");
                assert_eq!(&block[16 + run * 32..][..32], &single[4..], "group {group} run {run} codes");
            }
        }
    }

    #[test]
    fn every_encodable_codec_layout_round_trips_within_its_stated_error() {
        let layouts = encodable_layouts();
        let codecs: Vec<Codec> = layouts.iter().map(|(codec, _)| *codec).collect();
        let expected = [
            Codec::Q8_0,
            Codec::Q4_0,
            Codec::Q5_0,
            Codec::Q8_1,
            Codec::Q8K,
            Codec::Tq10,
            Codec::Tq20,
            Codec::Mxfp4,
            Codec::Nvfp4,
            Codec::Q1_0,
            Codec::Q2_0,
            Codec::Fp6E3M2,
            Codec::Pq2_0,
            Codec::Ptq1_0,
            Codec::Q8_1Grouped,
        ];
        assert_eq!(
            codecs, expected,
            "the codecs one scale per run encodes, in wire-tag order"
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
    fn encoding_a_decoded_integer_run_reproduces_the_same_bytes() {
        for (codec, layout) in encodable_layouts() {
            if !integer_code(&layout) {
                continue;
            }
            let values = inputs(&layout, 0xfeed ^ u64::from(layout.block_bytes));
            let first = encode(&layout, &values).expect("an encodable layout encodes");
            let second =
                encode(&layout, &decoded(&layout, &first)).expect("the decoded run encodes");
            assert_eq!(first, second, "{codec:?} is not idempotent on its own grid");
        }
    }

    #[test]
    fn encoding_a_decoded_float_code_run_keeps_every_value() {
        // a power-of-two scale can step down on a second pass, so the bytes are not fixed; the
        // values are, because every element of the grid stays representable at any such scale
        let mut checked = 0;
        for (codec, layout) in encodable_layouts() {
            if integer_code(&layout) {
                continue;
            }
            let values = inputs(&layout, 0xbeef ^ u64::from(layout.block_bytes));
            let once = decoded(
                &layout,
                &encode(&layout, &values).expect("an encodable layout encodes"),
            );
            let twice = decoded(&layout, &encode(&layout, &once).expect("the decoded run encodes"));
            assert_eq!(once, twice, "{codec:?} drifted on a second pass");
            checked += 1;
        }
        assert_eq!(checked, 3, "pinned: formats whose element code is a float");
    }

    #[test]
    fn layouts_with_sub_blocks_or_a_min_are_a_typed_error() {
        // TQ1_0 is not here: it has neither a sub-block nor a stored min, and its base-3 packing
        // encodes through the shared writer now (proxima-numeric pins that split per format)
        let refused = [BlockLayout::Q4_1, BlockLayout::Q4_K, BlockLayout::Q6_K];
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
