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
    use proxima_numeric::SignEncoding;
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

    /// An unsigned code centred on its bias (`Q4_0`, `Q5_0`): the layouts llama.cpp fits to the
    /// signed maximum rather than the peak magnitude.
    fn centred_offset_code(layout: &BlockLayout) -> bool {
        let bits = u32::from(layout.element.low.bits)
            + layout.element.high.map_or(0, |plane| u32::from(plane.bits));
        integer_code(layout)
            && matches!(layout.element.code.sign, SignEncoding::Absent)
            && i32::from(layout.element.bias) == 1i32 << (bits - 1)
    }

    /// The scale the encoder will store for this block peak, read back through the scale layout.
    fn scale_of(layout: &BlockLayout, peak: f32) -> f32 {
        let reach = block_layout_encode::reach(layout).expect("an encodable layout has a reach");
        let code = proxima_numeric::float_layout_codec::encode(peak / reach, layout.scale.layout);
        proxima_numeric::float_layout_codec::decode(code, layout.scale.layout)
    }

    fn stated_error(layout: &BlockLayout, peak: f32) -> f32 {
        if centred_offset_code(layout) {
            // llama's q4_0 and q5_0 scale by the signed maximum over the bias, so the opposite
            // end of the range saturates one step short: the error is a whole step, not half
            peak / f32::from(layout.element.bias) + 2.0 * peak * F16_EPSILON
        } else if integer_code(layout) {
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

    /// A 128-value activation row: gaussian-ish with two outlier channels per 64, the shape of a
    /// real key row, from splitmix64 seeded 0x6b76_5f72_6f77_0001 and a sum of twelve uniforms.
    fn activation_row() -> Vec<f32> {
        let mut state = 0x6b76_5f72_6f77_0001u64;
        let mut unit = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut mixed = state;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((mixed ^ (mixed >> 31)) >> 40) as f32 / (1u64 << 24) as f32
        };
        (0..128)
            .map(|index| {
                let gaussian = (0..12).map(|_| unit()).sum::<f32>() - 6.0;
                if matches!(index % 64, 5 | 37) { gaussian * 24.0 } else { gaussian }
            })
            .collect()
    }

    /// `quantize_row_q8_1_ref` of `activation_row()`, 4 blocks of 36 bytes: the bytes llama.cpp
    /// itself writes. Captured from libggml-base.dylib (built 2026-09-06 14:34 local, from the
    /// llama.cpp checkout at f1ea206218210afb913ae2f5d2c51faed35915da) called through dlopen by the
    /// xcheck-enc harness, never from a proxima port.
    const LLAMA_Q8_1_ROW: [u8; 144] = [
        0xcb, 0x2c, 0x3e, 0xc5, 0xf8, 0xee, 0x07, 0x08, 0xef, 0x81, 0xfb, 0x0f, 0x03, 0xec, 0xe1, 0x10,
        0xff, 0xfb, 0x08, 0x06, 0x0f, 0x1b, 0x02, 0xf3, 0xf6, 0x03, 0xfe, 0xff, 0x29, 0x06, 0x18, 0x07,
        0x08, 0xf3, 0x0f, 0xf6, 0x79, 0x33, 0x4c, 0xcf, 0x06, 0x01, 0xfd, 0x04, 0xff, 0x81, 0xf3, 0x03,
        0xf9, 0xfc, 0x03, 0x03, 0x00, 0x01, 0x04, 0x06, 0x02, 0xfa, 0x08, 0x08, 0x02, 0xfa, 0x00, 0x00,
        0x0a, 0xfc, 0xfc, 0x00, 0x01, 0x00, 0xfb, 0xf9, 0x9e, 0x2e, 0x11, 0xc9, 0x0b, 0x15, 0xf3, 0xff,
        0xfe, 0x81, 0xfb, 0xff, 0x04, 0xf5, 0xf0, 0x06, 0xfe, 0x0d, 0xef, 0x0c, 0x06, 0xfe, 0x00, 0x01,
        0x17, 0x09, 0x06, 0xe5, 0x09, 0xfc, 0xf7, 0x03, 0x09, 0xf3, 0x01, 0x12, 0x58, 0x31, 0x64, 0xce,
        0x01, 0xf8, 0xf5, 0x06, 0x07, 0x81, 0xff, 0x03, 0x01, 0xfe, 0x03, 0xf8, 0x00, 0x00, 0xfd, 0xff,
        0xf4, 0x06, 0x00, 0x0c, 0x04, 0xff, 0xf5, 0x08, 0xf9, 0xfc, 0xfb, 0x0b, 0x05, 0xfe, 0xfb, 0xf4,
    ];

    #[test]
    fn q8_1_activation_row_bytes_equal_llama_cpp_quantize_row_q8_1_ref() {
        let row = activation_row();
        let values: Vec<f32> = (0..ROWS).flat_map(|_| row.iter().copied()).collect();

        let packed = encode(&BlockLayout::Q8_1, &values).expect("q8_1 encodes a finite row");

        let expected: Vec<u8> = (0..ROWS).flat_map(|_| LLAMA_Q8_1_ROW).collect();
        assert_eq!(packed, expected);
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
                // the grouped layout is encoded by the shared walker, which divides by the rounded
                // scale; the single layout by llama's rule, which multiplies by the inverse of the
                // unrounded one: same scale, codes within one step, and a stored sum that follows them
                assert_eq!(&block[run * 4..][..2], &single[..2], "group {group} run {run} scale");
                for (grouped_code, single_code) in block[16 + run * 32..][..32].iter().zip(&single[4..]) {
                    let step = i16::from(*grouped_code as i8) - i16::from(*single_code as i8);
                    assert!(step.abs() <= 1, "group {group} run {run} codes differ by {step}");
                }
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

    /// Q8_1's `s` is `f16(d * sum(q))` from the scale before it is rounded to f16, as llama.cpp
    /// computes it; a re-encode from the decoded grid only has the rounded scale, so that field
    /// may move by one ulp while every other byte is fixed.
    fn without_stored_sums(codec: Codec, packed: &[u8]) -> Vec<u8> {
        let mut bytes = packed.to_vec();
        if codec == Codec::Q8_1 {
            for block in bytes.chunks_mut(36) {
                block[2..4].fill(0);
            }
        }
        bytes
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
            assert_eq!(
                without_stored_sums(codec, &first),
                without_stored_sums(codec, &second),
                "{codec:?} is not idempotent on its own grid"
            );
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
    fn sub_block_layouts_are_a_typed_error() {
        // TQ1_0 is not here: it has neither a sub-block nor a stored min, and its base-3 packing
        // encodes through the shared writer now (proxima-numeric pins that split per format)
        let refused = [BlockLayout::Q5_K, BlockLayout::Q6_K];
        for layout in refused {
            let error =
                encode(&layout, &inputs(&layout, 7)).expect_err("sub-block scales are not fitted");
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

    /// proxima-numeric's PROXNUM-193 oracle input: x[j] = ((j * 37 mod 64) - 20) / 16.
    fn oracle_input(count: usize) -> Vec<f32> {
        (0..count).map(|index| ((index * 37 % 64) as f32 - 20.0) / 16.0).collect()
    }

    #[test]
    fn stored_min_layouts_quantize_and_decode_within_half_a_step_per_element() {
        for (layout, levels) in [(BlockLayout::Q4_1, 15.0f32), (BlockLayout::Q5_1, 31.0f32)] {
            let width = usize::from(layout.elements);
            let values = oracle_input(ROWS as usize * row_width(&layout) as usize);
            let packed = encode(&layout, &values).expect("a stored-min layout encodes from its description");
            let back = decoded(&layout, &packed);

            assert_eq!(back.len(), values.len(), "{layout:?}: every element decodes");
            for (block, (source, restored)) in values.chunks(width).zip(back.chunks(width)).enumerate() {
                let low = source.iter().copied().fold(f32::INFINITY, f32::min);
                let high = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let step = (high - low) / levels;
                // d and min are stored as f16, so the half-step bound widens by their rounding
                let bound = step / 2.0 + step * levels * F16_EPSILON / 2.0 + high.abs().max(low.abs()) * F16_EPSILON;
                for (want, got) in source.iter().zip(restored) {
                    assert!((want - got).abs() <= bound, "{layout:?} block {block}: {want} vs {got}, bound {bound}");
                }
            }
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
