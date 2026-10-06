use alloc::string::ToString;

use proxima_gguf::{GgmlType as WireType, GgufModel, TensorPayload, write_complete};

use super::*;
use crate::error::InteropError;

fn dims(values: &[u64]) -> arrayvec::ArrayVec<u64, { proxima_gguf::tensor::MAX_DIMS }> {
    values.iter().copied().collect()
}

/// The decisive proof for this file's own fix: a raw-packed `F32` matmul
/// weight bound through [`bind_matmul_weight_as`] must land in
/// [`BoundWeights::owned`], transposed, and produce the exact same
/// matmul result as an independent hand computation over the tensor's
/// own on-disk bytes -- run through the real
/// [`proxima_tensor::cpu::evaluate_quantized_named`] evaluation a forward
/// program actually uses, not merely compared byte-for-byte against the
/// dequantize-then-transpose path.
///
/// `out_dim=4`/`in_dim=6` are deliberately asymmetric: this is exactly
/// what let the bug this test guards against hide behind a
/// one-channel or square fixture (see this module's own dense-vs-packed
/// history) -- a transpose of a square or single-row buffer can
/// coincidentally read back correctly, or silently permute symmetric
/// data, so it proves nothing. With `out_dim != in_dim`, a buffer
/// addressed through the wrong axis order reads flatly wrong values.
#[cfg(feature = "std")]
#[test]
fn raw_packed_f32_matmul_weight_matches_an_independent_hand_computed_matmul() {
    let out_dim = 4usize;
    let in_dim = 6usize;
    // GGUF's own on-disk convention: `out_dim` rows, each a contiguous
    // run of `in_dim` elements -- weight(out, in) = out*10 + in, distinct
    // per element so a scrambled read is detectable.
    let mut on_disk = vec![0.0f32; out_dim * in_dim];
    for out_index in 0..out_dim {
        for in_index in 0..in_dim {
            on_disk[out_index * in_dim + in_index] = (out_index * 10 + in_index) as f32;
        }
    }
    let bytes: Vec<u8> = on_disk
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();

    let model = GgufModel {
        version: 3,
        metadata: Vec::new(),
        tensors: vec![TensorPayload {
            name: "blk.0.ffn_gate_inp.weight".to_string(),
            dims: dims(&[in_dim as u64, out_dim as u64]),
            ggml_type: WireType::F32,
            data: &bytes,
        }],
    };
    let file_bytes =
        write_complete(&model).expect("writes gguf with an asymmetric f32 matmul weight");
    let parsed = proxima_gguf::pipe::parse_complete(&file_bytes)
        .expect("parses gguf with an asymmetric f32 matmul weight");

    let mut state = BoundWeights {
        resident_bytes: 0,
        owned: Vec::new(),
        packed: Vec::new(),
        packed_owned: Vec::new(),
        precision: &[],
    };
    bind_matmul_weight_as(
        &parsed,
        &file_bytes,
        "blk.0.ffn_gate_inp.weight",
        "gate".to_string(),
        out_dim,
        in_dim,
        &mut state,
    )
    .expect("binds the asymmetric f32 matmul weight");
    assert!(
        state.packed.is_empty(),
        "an F32 matmul weight must never take the raw-packed path -- nothing downstream corrects its layout"
    );
    assert_eq!(
        state.owned.len(),
        1,
        "the F32 matmul weight must land in the transposed owned path"
    );
    let bound_weight = &state.owned[0].1;

    // Deliberately NOT symmetric around zero (an earlier draft used
    // `index - 2.5`, whose sum over `0..6` is exactly zero -- that
    // silently canceled every `out_dim`-dependent term below and made
    // all four outputs identical regardless of which axis order the
    // buffer was actually read in, a vacuous test that would pass under
    // a transpose too. `index + 1` sums to a nonzero, out-axis-coupled
    // value instead.
    let activation: Vec<f32> = (0..in_dim).map(|index| (index as f32) + 1.0).collect();

    let mut program: Vec<proxima_tensor::op::Op> = Vec::new();
    let activation_node = proxima_tensor::op::append(
        &mut program,
        proxima_tensor::op::Op::Input {
            dtype: proxima_tensor::dtype::DType::Float32,
            shape: alloc::vec![proxima_tensor::op::Extent::Static(in_dim as u32)],
            name: Some("activation".to_string()),
        },
    );
    let weight_node = proxima_tensor::op::append(
        &mut program,
        proxima_tensor::op::Op::Input {
            dtype: proxima_tensor::dtype::DType::Float32,
            shape: alloc::vec![
                proxima_tensor::op::Extent::Static(in_dim as u32),
                proxima_tensor::op::Extent::Static(out_dim as u32)
            ],
            name: Some("gate".to_string()),
        },
    );
    let product = proxima_tensor::op::append(
        &mut program,
        proxima_tensor::op::Op::Elementwise {
            dtype: proxima_tensor::dtype::DType::Float32,
            body: proxima_tensor::op::ScalarOp::Multiply,
            operands: alloc::vec![
                (
                    activation_node,
                    proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                        2,
                        &[1]
                    ))
                ),
                (
                    weight_node,
                    proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                        2,
                        &[1, 0]
                    ))
                ),
            ],
            name: None,
        },
    );
    let logits = proxima_tensor::op::append(
        &mut program,
        proxima_tensor::op::Op::Reduce(proxima_tensor::op::Reduce {
            dtype: proxima_tensor::dtype::DType::Float32,
            body: proxima_tensor::op::ScalarOp::Add,
            init: proxima_tensor::op::ReduceInit::Zero,
            operand: product,
            in_map: proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                2,
                &[0, 1],
            )),
            out_map: proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                2,
                &[0],
            )),
            keep: proxima_tensor::op::Keep::Reduce,
            name: Some("logits".to_string()),
        }),
    );

    let named = [
        (
            "activation",
            proxima_tensor::cpu::QuantizedBlock::Float32(activation.as_slice()),
        ),
        (
            "gate",
            proxima_tensor::cpu::QuantizedBlock::Float32(bound_weight.as_slice()),
        ),
    ];
    let evaluated =
        proxima_tensor::cpu::evaluate_quantized_named(&program, &[], &named, &[logits])
            .expect("evaluate the bound matmul weight through the real interpreter");
    let ours = evaluated.root();

    let mut oracle = vec![0.0f32; out_dim];
    for (out_index, logit) in oracle.iter_mut().enumerate() {
        let mut accumulator = 0.0f32;
        for in_index in 0..in_dim {
            accumulator += activation[in_index] * on_disk[out_index * in_dim + in_index];
        }
        *logit = accumulator;
    }

    std::println!("raw_packed_f32_matmul ours={ours:?} oracle={oracle:?}");
    for (out_index, (found, wanted)) in ours.iter().zip(&oracle).enumerate() {
        let diff = (found - wanted).abs();
        assert!(
            diff < 1e-4,
            "output {out_index}: found={found} wanted={wanted} diff={diff}"
        );
    }
}

/// `[rows, k]` for [`q8_0_matmul_weight_binds_packed_and_matches_a_dequantized_oracle`]
/// and its mutation companion below -- `k` a multiple of
/// [`q8_0::QK8_0`] (32) so every row's own blocks stay row-aligned (no
/// block straddles two rows), matching [`proxima_tensor::cpu::matmul_q8_0_f32`]'s
/// own per-row block assumption. `rows != k` for the same
/// axis-order-detection reason the `F32` test above documents.
#[cfg(feature = "std")]
const Q8_0_TEST_ROWS: usize = 3;
#[cfg(feature = "std")]
const Q8_0_TEST_K: usize = 64;

/// Row-major `[rows, k]` weight bytes, real `Q8_0` blocks (never a
/// hand-built buffer) via [`q8_0::quantize`] -- deterministic,
/// non-degenerate per-element values so no two elements collide.
#[cfg(feature = "std")]
fn quantized_q8_0_weight_bytes() -> (alloc::vec::Vec<f32>, alloc::vec::Vec<u8>) {
    let on_disk: alloc::vec::Vec<f32> = (0..Q8_0_TEST_ROWS * Q8_0_TEST_K)
        .map(|index| ((index % 41) as f32 - 20.0) / 8.0)
        .collect();
    let mut bytes = alloc::vec![0u8; (on_disk.len() / q8_0::QK8_0) * q8_0::BLOCK_BYTES];
    q8_0::quantize(&on_disk, &mut bytes).expect("real q8_0 encoder quantizes this fixture");
    (on_disk, bytes)
}

/// Builds the `[rows, 1] x [rows, k] -> [rows, 1]` quantized-matmul
/// program [`proxima_tensor::cpu::run_reduce_quantized`]'s own packed
/// dispatch recognizes -- the same op shape `proxima_tensor::cpu`'s own
/// `quantized_matmul_program` test helper builds (that helper is
/// private to `cpu.rs`'s own test module, so this is a same-shape,
/// independently written copy, not a shared function), rebuilt here
/// through the named-`Op::Input` [`gguf_tensor_as_packed_block`]'s own
/// callers actually use.
#[cfg(feature = "std")]
fn q8_0_matmul_program() -> (Vec<proxima_tensor::op::Op>, proxima_tensor::op::NodeId) {
    let mut program: Vec<proxima_tensor::op::Op> = Vec::new();
    let weight_node = proxima_tensor::op::append(
        &mut program,
        proxima_tensor::op::Op::Input {
            dtype: proxima_tensor::dtype::DType::UInt8,
            shape: alloc::vec![
                proxima_tensor::op::Extent::Static(Q8_0_TEST_ROWS as u32),
                proxima_tensor::op::Extent::Static(Q8_0_TEST_K as u32)
            ],
            name: Some("weight".to_string()),
        },
    );
    let activation_node = proxima_tensor::op::append(
        &mut program,
        proxima_tensor::op::Op::Input {
            dtype: proxima_tensor::dtype::DType::Float32,
            shape: alloc::vec![
                proxima_tensor::op::Extent::Static(Q8_0_TEST_K as u32),
                proxima_tensor::op::Extent::Static(1)
            ],
            name: Some("activation".to_string()),
        },
    );
    let product = proxima_tensor::op::append(
        &mut program,
        proxima_tensor::op::Op::Elementwise {
            dtype: proxima_tensor::dtype::DType::Float32,
            body: proxima_tensor::op::ScalarOp::Multiply,
            operands: alloc::vec![
                (
                    weight_node,
                    proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                        3,
                        &[0, 2]
                    ))
                ),
                (
                    activation_node,
                    proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                        3,
                        &[2, 1]
                    ))
                ),
            ],
            name: None,
        },
    );
    let sum = proxima_tensor::op::append(
        &mut program,
        proxima_tensor::op::Op::Reduce(proxima_tensor::op::Reduce {
            dtype: proxima_tensor::dtype::DType::Float32,
            body: proxima_tensor::op::ScalarOp::Add,
            init: proxima_tensor::op::ReduceInit::Zero,
            operand: product,
            in_map: proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                3,
                &[0, 1, 2],
            )),
            out_map: proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                3,
                &[0, 1],
            )),
            keep: proxima_tensor::op::Keep::Reduce,
            name: Some("q8_0_matmul".to_string()),
        }),
    );
    (program, sum)
}

/// This module's fix, proved directly: a `Q8_0` matmul weight must bind
/// through [`gguf_tensor_as_packed_block`] into [`BoundWeights::packed`]
/// zero-copy -- never fall through to [`gguf_tensor_as_f32`]'s owned
/// dequantize path -- and the real
/// [`proxima_tensor::cpu::matmul_q8_0_f32`] kernel driven off that
/// packed buffer must produce the exact same output as an independent
/// oracle: [`q8_0::dequantize`] applied to the SAME packed bytes,
/// matmul'd by hand. The oracle dequantizes the packed bytes rather than
/// the pre-quantization `f32` source, because `Q8_0` quantization is
/// itself lossy -- comparing against the pre-quantization values would
/// conflate this bind wiring's own correctness with `Q8_0`'s codec
/// accuracy (already proved in `proxima_gguf::quant::q8_0`'s own tests).
#[cfg(feature = "std")]
#[test]
fn q8_0_matmul_weight_binds_packed_and_matches_a_dequantized_oracle() {
    let (_on_disk, packed_bytes) = quantized_q8_0_weight_bytes();

    let model = GgufModel {
        version: 3,
        metadata: Vec::new(),
        tensors: vec![TensorPayload {
            name: "blk.0.attn_q.weight".to_string(),
            dims: dims(&[Q8_0_TEST_K as u64, Q8_0_TEST_ROWS as u64]),
            ggml_type: WireType::Q8_0,
            data: &packed_bytes,
        }],
    };
    let file_bytes = write_complete(&model).expect("writes gguf with a real q8_0 weight");
    let parsed =
        proxima_gguf::pipe::parse_complete(&file_bytes).expect("parses q8_0 weight gguf");

    let mut state = BoundWeights {
        resident_bytes: 0,
        owned: Vec::new(),
        packed: Vec::new(),
        packed_owned: Vec::new(),
        precision: &[],
    };
    bind_matmul_weight_as(
        &parsed,
        &file_bytes,
        "blk.0.attn_q.weight",
        "weight".to_string(),
        Q8_0_TEST_ROWS,
        Q8_0_TEST_K,
        &mut state,
    )
    .expect("binds the q8_0 matmul weight");
    assert!(
        state.owned.is_empty(),
        "a q8_0 matmul weight must take the zero-copy packed path, never the owned dequantize \
         fallback -- this is the exact defect this change fixes"
    );
    assert_eq!(state.packed.len(), 1, "exactly one packed weight bound");
    let bound_bytes = match &state.packed[0].1 {
        proxima_tensor::cpu::QuantizedBlock::Packed { codec: Codec::Q8_0, bytes } => *bytes,
        other => panic!("expected a QuantizedBlock::Packed with Codec::Q8_0, found {other:?}"),
    };
    assert_eq!(
        bound_bytes,
        packed_bytes.as_slice(),
        "the packed path must borrow the exact on-disk q8_0 bytes, no copy"
    );

    let activation: Vec<f32> = (0..Q8_0_TEST_K)
        .map(|index| (index as f32) - 32.0)
        .collect();
    let (program, sum) = q8_0_matmul_program();
    let named = [
        (
            "weight",
            proxima_tensor::cpu::QuantizedBlock::Packed { codec: Codec::Q8_0, bytes: bound_bytes },
        ),
        (
            "activation",
            proxima_tensor::cpu::QuantizedBlock::Float32(activation.as_slice()),
        ),
    ];
    let evaluated =
        proxima_tensor::cpu::evaluate_quantized_named(&program, &[], &named, &[sum])
            .expect("evaluate the bound q8_0 packed weight through the real interpreter");
    let ours = evaluated.root();

    let mut dequantized_weight = alloc::vec![0.0f32; Q8_0_TEST_ROWS * Q8_0_TEST_K];
    q8_0::dequantize(&packed_bytes, &mut dequantized_weight)
        .expect("dequantize the same packed bytes for the oracle");
    let mut oracle = alloc::vec![0.0f32; Q8_0_TEST_ROWS];
    for (row, logit) in oracle.iter_mut().enumerate() {
        let mut accumulator = 0.0f32;
        for column in 0..Q8_0_TEST_K {
            accumulator += activation[column] * dequantized_weight[row * Q8_0_TEST_K + column];
        }
        *logit = accumulator;
    }

    std::println!("q8_0_matmul ours={ours:?} oracle={oracle:?}");
    for (row, (found, wanted)) in ours.iter().zip(&oracle).enumerate() {
        let diff = (found - wanted).abs();
        assert!(
            diff < 1e-2,
            "row {row}: found={found} wanted={wanted} diff={diff}"
        );
    }
}

/// Mutation companion to
/// [`q8_0_matmul_weight_binds_packed_and_matches_a_dequantized_oracle`]:
/// runs the exact same bind-then-evaluate pipeline, but flips one packed
/// byte (inside a block's `qs` region, not its `d` scale header) before
/// binding, so the packed path decodes a deliberately wrong value. Then
/// asserts the real kernel's output on the corrupted bytes diverges from
/// the clean oracle beyond the previous test's own `1e-2` tolerance --
/// proving that tolerance is tight enough to actually catch a wrong
/// decode, not so loose the equivalence check above is vacuous.
#[cfg(feature = "std")]
#[test]
fn q8_0_matmul_weight_packed_path_is_sensitive_to_a_corrupted_byte() {
    let (_on_disk, clean_bytes) = quantized_q8_0_weight_bytes();
    let mut corrupted_bytes = clean_bytes.clone();
    // second block's 9th `qs` byte -- decoded element 40, whose
    // activation coefficient (`40 - 32 = 8`) is far from zero, unlike
    // element 32 (the second block's first element), whose activation
    // coefficient is exactly zero and would mask any corruption there.
    let corrupted_index = q8_0::BLOCK_BYTES + 2 + 8;
    // flips the signed byte's sign bit -- guarantees a large jump in the
    // decoded value regardless of what the original byte happened to be,
    // unlike a smaller XOR mask that can land near the original value.
    corrupted_bytes[corrupted_index] = corrupted_bytes[corrupted_index].wrapping_add(128);

    let mut clean_weight = alloc::vec![0.0f32; Q8_0_TEST_ROWS * Q8_0_TEST_K];
    q8_0::dequantize(&clean_bytes, &mut clean_weight)
        .expect("dequantize the clean packed bytes for the oracle");
    let activation: Vec<f32> = (0..Q8_0_TEST_K)
        .map(|index| (index as f32) - 32.0)
        .collect();
    let mut clean_oracle = alloc::vec![0.0f32; Q8_0_TEST_ROWS];
    for (row, logit) in clean_oracle.iter_mut().enumerate() {
        let mut accumulator = 0.0f32;
        for column in 0..Q8_0_TEST_K {
            accumulator += activation[column] * clean_weight[row * Q8_0_TEST_K + column];
        }
        *logit = accumulator;
    }

    let model = GgufModel {
        version: 3,
        metadata: Vec::new(),
        tensors: vec![TensorPayload {
            name: "blk.0.attn_q.weight".to_string(),
            dims: dims(&[Q8_0_TEST_K as u64, Q8_0_TEST_ROWS as u64]),
            ggml_type: WireType::Q8_0,
            data: &corrupted_bytes,
        }],
    };
    let file_bytes = write_complete(&model).expect("writes gguf with a corrupted q8_0 weight");
    let parsed = proxima_gguf::pipe::parse_complete(&file_bytes)
        .expect("parses corrupted q8_0 weight gguf");

    let mut state = BoundWeights {
        resident_bytes: 0,
        owned: Vec::new(),
        packed: Vec::new(),
        packed_owned: Vec::new(),
        precision: &[],
    };
    bind_matmul_weight_as(
        &parsed,
        &file_bytes,
        "blk.0.attn_q.weight",
        "weight".to_string(),
        Q8_0_TEST_ROWS,
        Q8_0_TEST_K,
        &mut state,
    )
    .expect("binds the corrupted q8_0 matmul weight");
    let bound_bytes = match &state.packed[0].1 {
        proxima_tensor::cpu::QuantizedBlock::Packed { codec: Codec::Q8_0, bytes } => *bytes,
        other => panic!("expected a QuantizedBlock::Packed with Codec::Q8_0, found {other:?}"),
    };

    let (program, sum) = q8_0_matmul_program();
    let named = [
        (
            "weight",
            proxima_tensor::cpu::QuantizedBlock::Packed { codec: Codec::Q8_0, bytes: bound_bytes },
        ),
        (
            "activation",
            proxima_tensor::cpu::QuantizedBlock::Float32(activation.as_slice()),
        ),
    ];
    let evaluated =
        proxima_tensor::cpu::evaluate_quantized_named(&program, &[], &named, &[sum])
            .expect("evaluate the corrupted q8_0 packed weight through the real interpreter");
    let corrupted_result = evaluated.root();

    std::println!(
        "q8_0_corrupted corrupted={corrupted_result:?} clean_oracle={clean_oracle:?}"
    );
    let max_diff = corrupted_result
        .iter()
        .zip(&clean_oracle)
        .map(|(found, wanted)| (found - wanted).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-2,
        "a corrupted qs byte must move the decoded matmul output past the equivalence \
         test's own 1e-2 tolerance, or that tolerance cannot detect a wrong decode: \
         max_diff={max_diff}"
    );
}

/// The defect this signature change fixes, proved directly: a decoded
/// buffer whose length disagrees with `expert_count * out_dim * in_dim`
/// (exactly what a GGUF file with a mismatched `general.expert_count`
/// hparam would hand this function) used to slice past the buffer's
/// end mid-transpose. It must now return a typed error instead.
#[cfg(feature = "std")]
#[test]
fn expert_stack_length_mismatch_is_a_typed_error_not_a_panic() {
    let expert_count = 4;
    let out_dim = 8;
    let in_dim = 8;
    let one_expert_short = vec![0.0f32; (expert_count - 1) * out_dim * in_dim];

    let result = transpose_expert_stack(
        &one_expert_short,
        "blk.0.ffn_gate_exps.weight",
        expert_count,
        out_dim,
        in_dim,
    );

    assert!(
        matches!(result, Err(InteropError::MoeExpertShapeMismatch { .. })),
        "a short decoded buffer must surface as a typed error, got {result:?}"
    );
}

/// Same proof, for the plain (non-MoE) dense-weight transpose: a
/// decoded buffer whose length disagrees with `out_dim * in_dim` used
/// to trip `assert_eq!` (a panic) instead of returning a typed error.
#[cfg(feature = "std")]
#[test]
fn dense_weight_length_mismatch_is_a_typed_error_not_a_panic() {
    let out_dim = 8;
    let in_dim = 8;
    let too_short = vec![0.0f32; out_dim * in_dim - 1];

    let result = transpose_out_in_to_in_out(&too_short, "output.weight", out_dim, in_dim);

    assert!(
        matches!(result, Err(InteropError::DenseWeightShapeMismatch { .. })),
        "a short decoded buffer must surface as a typed error, got {result:?}"
    );
}

/// A round-tripped `F32` tensor comes back byte-identical as `f32`,
/// not merely "close" -- reinterpretation, not conversion.
#[test]
fn f32_tensor_reinterprets_bytes_exactly() {
    let values = [1.0f32, -2.5, 3.25, 0.0];
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let model = GgufModel {
        version: 3,
        metadata: Vec::new(),
        tensors: vec![TensorPayload {
            name: "weights".to_string(),
            dims: dims(&[4]),
            ggml_type: WireType::F32,
            data: bytes.as_slice(),
        }],
    };
    let file_bytes = write_complete(&model).expect("writes gguf");
    let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses gguf");

    let decoded =
        gguf_tensor_as_f32(&parsed, &file_bytes, "weights").expect("bind f32 tensor by name");
    assert_eq!(decoded, values);
}

/// A `Q4_K` tensor decodes through the crate's own dequantizer and
/// matches an independent hand computation of the block's `x =
/// d*sc*q - dmin*m` formula for the one nonzero probe element this
/// fixture packs.
#[cfg(feature = "std")]
#[test]
fn q4_k_tensor_dequantizes_through_bind_by_name() {
    let mut block = [0u8; q4_k::BLOCK_BYTES];
    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes()); // d
    block[2..4].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes()); // dmin
    // sub_block 0: scale code 3, min code 61 (sub_block < 4 packing).
    block[4] = 3;
    block[8] = 61;
    block[16] = 0x07; // qs[0] low nibble = 7 -> element 0 of sub_block 0

    let model = GgufModel {
        version: 3,
        metadata: Vec::new(),
        tensors: vec![TensorPayload {
            name: "blk.0.ffn_gate.weight".to_string(),
            dims: dims(&[q4_k::QK_K as u64]),
            ggml_type: WireType::Q4_K,
            data: &block,
        }],
    };
    let file_bytes = write_complete(&model).expect("writes quantized gguf");
    let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses quantized gguf");

    let decoded = gguf_tensor_as_f32(&parsed, &file_bytes, "blk.0.ffn_gate.weight")
        .expect("bind q4_k tensor by name");
    assert_eq!(decoded.len(), q4_k::QK_K);
    // element 0: d*sc*q - dmin*m = 1.0*3.0*7.0 - 0.5*61.0 = -9.5
    assert!(
        (decoded[0] - (-9.5)).abs() < 1e-6,
        "decoded[0]={}",
        decoded[0]
    );
    // every other element in sub_block 0 shares scale/min with q=0.
    assert!(
        (decoded[1] - (-30.5)).abs() < 1e-6,
        "decoded[1]={}",
        decoded[1]
    );
}

#[test]
fn unknown_name_errors_instead_of_panicking() {
    let model = GgufModel {
        version: 3,
        metadata: Vec::new(),
        tensors: Vec::new(),
    };
    let file_bytes = write_complete(&model).expect("writes empty gguf");
    let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses empty gguf");

    let outcome = gguf_tensor_as_f32(&parsed, &file_bytes, "missing");
    assert!(matches!(outcome, Err(InteropError::UnknownTensor { .. })));
}

#[test]
fn unrepresentable_ggml_type_errors_instead_of_misreading_bytes() {
    let data = [0u8; 17]; // one MXFP4 block; registered, but no decoder is implemented
    let model = GgufModel {
        version: 3,
        metadata: Vec::new(),
        tensors: vec![TensorPayload {
            name: "blk.0.attn_q.weight".to_string(),
            dims: dims(&[32]),
            ggml_type: WireType::Mxfp4,
            data: &data,
        }],
    };
    let file_bytes = write_complete(&model).expect("writes mxfp4 gguf");
    let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses mxfp4 gguf");

    let outcome = gguf_tensor_as_f32(&parsed, &file_bytes, "blk.0.attn_q.weight");
    assert!(matches!(
        outcome,
        Err(InteropError::UnrepresentableGgmlType { .. })
    ));
}

/// [`architecture_from_metadata`] against a synthetic checkpoint whose
/// metadata carries every `{architecture}.*` key by hand -- proves the
/// derivation reads real keys rather than falling back to invented
/// defaults, and that `vocab` comes from `token_embd.weight`'s own
/// shape, not a metadata key (this fixture writes none).
#[test]
fn architecture_from_metadata_reads_real_keys_and_derives_vocab_from_tensor_shape() {
    use proxima_gguf::value::MetadataValue as Value;

    let embed_bytes = vec![0u8; 8 * 3 * 4]; // [embedding=8, vocab=3] f32
    let model = GgufModel {
        version: 3,
        metadata: vec![
            (
                "general.architecture".to_string(),
                Value::String("llama".to_string()),
            ),
            ("llama.embedding_length".to_string(), Value::U32(8)),
            ("llama.feed_forward_length".to_string(), Value::U32(32)),
            ("llama.attention.head_count".to_string(), Value::U32(2)),
            ("llama.attention.head_count_kv".to_string(), Value::U32(1)),
            ("llama.block_count".to_string(), Value::U32(4)),
            ("llama.rope.dimension_count".to_string(), Value::U32(4)),
        ],
        tensors: vec![TensorPayload {
            name: "token_embd.weight".to_string(),
            dims: dims(&[8, 3]),
            ggml_type: WireType::F32,
            data: &embed_bytes,
        }],
    };
    let file_bytes = write_complete(&model).expect("writes gguf with architecture metadata");
    let parsed = proxima_gguf::parse_complete(&file_bytes)
        .expect("parses gguf with architecture metadata");

    let architecture = architecture_from_metadata(&parsed)
        .expect("derive architecture from real metadata keys");
    assert_eq!(
        architecture,
        ModelHparams {
            vocab: 3,
            embedding: 8,
            feed_forward: 32,
            query_heads: 2,
            kv_heads: 1,
            kv_heads_by_layer: vec![1; 4],
            head_dim: 4,
            block_count: 4,
            expert_count: 0,
            expert_used_count: 0,
            rope_freq_base: proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
            rms_epsilon: RMS_EPSILON_DEFAULT,
            tied_embeddings: false,
            family: "llama".to_string(),
            sliding_rope: None,
        },
        "a checkpoint with no expert_count/expert_used_count key is dense: both fields must read as 0, \
         not error; a checkpoint with no rope.freq_base/layer_norm_rms_epsilon key must fall back \
         to the sizing-config default / RMS_EPSILON_DEFAULT"
    );
}

/// A mixture-of-experts checkpoint carries `{architecture}.expert_count`/
/// `{architecture}.expert_used_count` alongside every dense key --
/// `architecture_from_metadata` must read both rather than silently
/// treating the checkpoint as dense.
#[test]
fn architecture_from_metadata_reads_expert_count_when_present() {
    use proxima_gguf::value::MetadataValue as Value;

    let embed_bytes = vec![0u8; 8 * 3 * 4]; // [embedding=8, vocab=3] f32
    let model = GgufModel {
        version: 3,
        metadata: vec![
            (
                "general.architecture".to_string(),
                Value::String("llama".to_string()),
            ),
            ("llama.embedding_length".to_string(), Value::U32(8)),
            ("llama.feed_forward_length".to_string(), Value::U32(32)),
            ("llama.attention.head_count".to_string(), Value::U32(2)),
            ("llama.attention.head_count_kv".to_string(), Value::U32(1)),
            ("llama.block_count".to_string(), Value::U32(4)),
            ("llama.rope.dimension_count".to_string(), Value::U32(4)),
            ("llama.expert_count".to_string(), Value::U32(8)),
            ("llama.expert_used_count".to_string(), Value::U32(2)),
        ],
        tensors: vec![TensorPayload {
            name: "token_embd.weight".to_string(),
            dims: dims(&[8, 3]),
            ggml_type: WireType::F32,
            data: &embed_bytes,
        }],
    };
    let file_bytes = write_complete(&model).expect("writes gguf with moe metadata");
    let parsed =
        proxima_gguf::parse_complete(&file_bytes).expect("parses gguf with moe metadata");

    let architecture = architecture_from_metadata(&parsed)
        .expect("derive architecture from real metadata keys");
    assert_eq!(
        architecture.expert_count, 8,
        "expert_count must read the real metadata key, not default to 0"
    );
    assert_eq!(
        architecture.expert_used_count, 2,
        "expert_used_count must read the real metadata key, not default to 0"
    );
}

/// Root cause of `InteropError::MoeExpertShapeMismatch` on the real
/// qwen3moe 30B-A3B checkpoint (`~/.ollama/models/blobs/sha256-58574f2e..`):
/// its header declares `qwen3moe.feed_forward_length=6144` (a legacy /
/// unused dense value) alongside a *separate*
/// `qwen3moe.expert_feed_forward_length=768`, the real per-expert
/// projection width `blk.0.ffn_gate_exps.weight`'s own on-disk element
/// count (`201_326_592 = 128 * 768 * 2048`) agrees with. Before this
/// fix, `architecture_from_metadata` read only `feed_forward_length`
/// unconditionally, so `bind_moe_expert_weights` called
/// `transpose_expert_stack` with `out_dim=6144` instead of `768` --
/// `expected = 128 * 6144 * 2048 = 1_610_612_736`, 8x the tensor's real
/// element count, since `6144 / 768 == 8`. This fixture reproduces both
/// keys at that exact ratio and asserts `feed_forward` reads the
/// expert-specific key, not the dense one, once `expert_count != 0` --
/// `architecture_from_hf_config` (`hf_config.rs`) already makes this
/// same substitution for the HF-config/safetensors path.
#[test]
fn architecture_from_metadata_reads_expert_feed_forward_length_when_present() {
    use proxima_gguf::value::MetadataValue as Value;

    let embed_bytes = vec![0u8; 8 * 3 * 4]; // [embedding=8, vocab=3] f32
    let model = GgufModel {
        version: 3,
        metadata: vec![
            (
                "general.architecture".to_string(),
                Value::String("qwen3moe".to_string()),
            ),
            ("qwen3moe.embedding_length".to_string(), Value::U32(8)),
            ("qwen3moe.feed_forward_length".to_string(), Value::U32(64)),
            (
                "qwen3moe.expert_feed_forward_length".to_string(),
                Value::U32(8),
            ),
            ("qwen3moe.attention.head_count".to_string(), Value::U32(2)),
            (
                "qwen3moe.attention.head_count_kv".to_string(),
                Value::U32(1),
            ),
            ("qwen3moe.block_count".to_string(), Value::U32(4)),
            ("qwen3moe.rope.dimension_count".to_string(), Value::U32(4)),
            ("qwen3moe.expert_count".to_string(), Value::U32(128)),
            ("qwen3moe.expert_used_count".to_string(), Value::U32(8)),
        ],
        tensors: vec![TensorPayload {
            name: "token_embd.weight".to_string(),
            dims: dims(&[8, 3]),
            ggml_type: WireType::F32,
            data: &embed_bytes,
        }],
    };
    let file_bytes = write_complete(&model).expect("writes gguf with qwen3moe-shaped metadata");
    let parsed = proxima_gguf::parse_complete(&file_bytes)
        .expect("parses gguf with qwen3moe-shaped metadata");

    let architecture = architecture_from_metadata(&parsed)
        .expect("derive architecture from real qwen3moe-shaped metadata keys");
    assert_eq!(
        architecture.feed_forward, 8,
        "an expert checkpoint's feed_forward must read expert_feed_forward_length (8), not \
         the dense feed_forward_length (64), once expert_count is nonzero"
    );
}

/// A checkpoint absent `{architecture}.rope.dimension_count` entirely
/// (confirmed real on the 8B-A1B short-conv checkpoint, `bind.rs`'s own doc on
/// [`ModelHparams::head_dim`]) must derive `head_dim` as
/// `embedding / query_heads` rather than
/// [`InteropError::MissingMetadataKey`] -- this fixture's own
/// embedding=8, query_heads=2 implies head_dim=4, matching what this
/// same fixture's other tests declare explicitly via the key.
#[test]
fn architecture_from_metadata_derives_head_dim_when_rope_dimension_count_is_absent() {
    use proxima_gguf::value::MetadataValue as Value;

    let embed_bytes = vec![0u8; 8 * 3 * 4]; // [embedding=8, vocab=3] f32
    let model = GgufModel {
        version: 3,
        metadata: vec![
            (
                "general.architecture".to_string(),
                Value::String("llama".to_string()),
            ),
            ("llama.embedding_length".to_string(), Value::U32(8)),
            ("llama.feed_forward_length".to_string(), Value::U32(32)),
            ("llama.attention.head_count".to_string(), Value::U32(2)),
            ("llama.attention.head_count_kv".to_string(), Value::U32(1)),
            ("llama.block_count".to_string(), Value::U32(4)),
            // deliberately no llama.rope.dimension_count key
        ],
        tensors: vec![TensorPayload {
            name: "token_embd.weight".to_string(),
            dims: dims(&[8, 3]),
            ggml_type: WireType::F32,
            data: &embed_bytes,
        }],
    };
    let file_bytes = write_complete(&model).expect("writes gguf without rope.dimension_count");
    let parsed = proxima_gguf::parse_complete(&file_bytes)
        .expect("parses gguf without rope.dimension_count");

    let architecture = architecture_from_metadata(&parsed)
        .expect("absent rope.dimension_count must derive, not error");
    assert_eq!(
        architecture.head_dim, 4,
        "head_dim must derive as embedding(8) / query_heads(2)"
    );
}

/// `{architecture}.attention.head_count_kv` stored as a per-layer
/// [`proxima_gguf::value::MetadataArray`] whose entries all agree
/// (every real dense/uniform checkpoint that ever uses the array
/// encoding at all) must read as that one scalar, not error.
#[test]
fn architecture_from_metadata_reads_uniform_head_count_kv_array() {
    use proxima_gguf::value::MetadataArray;
    use proxima_gguf::value::MetadataValue as Value;

    let embed_bytes = vec![0u8; 8 * 3 * 4];
    let model = GgufModel {
        version: 3,
        metadata: vec![
            (
                "general.architecture".to_string(),
                Value::String("llama".to_string()),
            ),
            ("llama.embedding_length".to_string(), Value::U32(8)),
            ("llama.feed_forward_length".to_string(), Value::U32(32)),
            ("llama.attention.head_count".to_string(), Value::U32(2)),
            (
                "llama.attention.head_count_kv".to_string(),
                Value::Array(MetadataArray::I32(vec![1, 1, 1, 1])),
            ),
            ("llama.block_count".to_string(), Value::U32(4)),
            ("llama.rope.dimension_count".to_string(), Value::U32(4)),
        ],
        tensors: vec![TensorPayload {
            name: "token_embd.weight".to_string(),
            dims: dims(&[8, 3]),
            ggml_type: WireType::F32,
            data: &embed_bytes,
        }],
    };
    let file_bytes =
        write_complete(&model).expect("writes gguf with a uniform head_count_kv array");
    let parsed = proxima_gguf::parse_complete(&file_bytes)
        .expect("parses gguf with a uniform head_count_kv array");

    let architecture = architecture_from_metadata(&parsed)
        .expect("a uniform per-layer array must read as its one scalar");
    assert_eq!(architecture.kv_heads, 1);
}

/// `{architecture}.attention.head_count_kv` stored as a per-layer array
/// whose entries genuinely differ must remain configuration data in
/// [`ModelHparams`]. A uniform builder can still reject it through
/// [`ModelHparams::uniform_kv_heads`], but parsing must not erase
/// the locations or select a representative layer.
#[test]
fn architecture_from_metadata_preserves_a_heterogeneous_head_count_kv_array() {
    use proxima_gguf::value::MetadataArray;
    use proxima_gguf::value::MetadataValue as Value;

    let embed_bytes = vec![0u8; 8 * 3 * 4];
    let model = GgufModel {
        version: 3,
        metadata: vec![
            (
                "general.architecture".to_string(),
                Value::String("lfm2".to_string()),
            ),
            ("lfm2.embedding_length".to_string(), Value::U32(8)),
            ("lfm2.feed_forward_length".to_string(), Value::U32(32)),
            ("lfm2.attention.head_count".to_string(), Value::U32(2)),
            (
                "lfm2.attention.head_count_kv".to_string(),
                Value::Array(MetadataArray::I32(vec![0, 0, 8, 0, 8, 8])),
            ),
            ("lfm2.block_count".to_string(), Value::U32(6)),
            ("lfm2.rope.dimension_count".to_string(), Value::U32(4)),
        ],
        tensors: vec![TensorPayload {
            name: "token_embd.weight".to_string(),
            dims: dims(&[8, 3]),
            ggml_type: WireType::F32,
            data: &embed_bytes,
        }],
    };
    let file_bytes =
        write_complete(&model).expect("writes gguf with a heterogeneous head_count_kv array");
    let parsed = proxima_gguf::parse_complete(&file_bytes)
        .expect("parses gguf with a heterogeneous head_count_kv array");

    let architecture = architecture_from_metadata(&parsed)
        .expect("a heterogeneous per-layer configuration must parse");
    assert!(
        matches!(
            architecture.uniform_kv_heads(),
            Err(InteropError::HeterogeneousMetadataArray {
                distinct_values: 2,
                ..
            })
        ),
        "a uniform consumer must still reject a genuinely varying configuration"
    );
    assert_eq!(architecture.kv_heads, 0);
    assert_eq!(architecture.kv_heads_by_layer, vec![0, 0, 8, 0, 8, 8]);
}

/// A checkpoint declaring a non-`10_000.0` `{architecture}.rope.freq_base`
/// (Qwen3's real `1_000_000.0`, Llama 3's real `500_000.0`) must have
/// `architecture_from_metadata` read that value, not silently fall back
/// to a hardcoded default -- the exact defect this test is named for:
/// before `ModelHparams` carried a `rope_freq_base` field at all,
/// this assertion could not even be written, let alone pass, and every
/// production call site used a bare `10_000.0` constant regardless of
/// what a checkpoint declared.
#[proxima::test]
#[case::qwen3_real_freq_base(1_000_000.0)]
#[case::llama3_real_freq_base(500_000.0)]
async fn architecture_from_metadata_reads_the_real_rope_freq_base(#[case] freq_base: f32) {
    use proxima_gguf::value::MetadataValue as Value;

    let embed_bytes = vec![0u8; 8 * 3 * 4]; // [embedding=8, vocab=3] f32
    let model = GgufModel {
        version: 3,
        metadata: vec![
            (
                "general.architecture".to_string(),
                Value::String("llama".to_string()),
            ),
            ("llama.embedding_length".to_string(), Value::U32(8)),
            ("llama.feed_forward_length".to_string(), Value::U32(32)),
            ("llama.attention.head_count".to_string(), Value::U32(2)),
            ("llama.attention.head_count_kv".to_string(), Value::U32(1)),
            ("llama.block_count".to_string(), Value::U32(4)),
            ("llama.rope.dimension_count".to_string(), Value::U32(4)),
            ("llama.rope.freq_base".to_string(), Value::F32(freq_base)),
        ],
        tensors: vec![TensorPayload {
            name: "token_embd.weight".to_string(),
            dims: dims(&[8, 3]),
            ggml_type: WireType::F32,
            data: &embed_bytes,
        }],
    };
    let file_bytes =
        write_complete(&model).expect("writes gguf with a non-default rope.freq_base");
    let parsed = proxima_gguf::parse_complete(&file_bytes)
        .expect("parses gguf with a non-default rope.freq_base");

    let architecture = architecture_from_metadata(&parsed)
        .expect("derive architecture from real metadata keys");
    assert_eq!(
        architecture.rope_freq_base,
        freq_base,
        "rope_freq_base must read the checkpoint's own metadata key, never the {}-default",
        proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT
    );
    assert_ne!(
        architecture.rope_freq_base,
        proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
        "this case is only meaningful when the checkpoint's declared value differs from the default"
    );
}

/// The `rope.freq_base` key stored as `F64` on the wire (a legal, if
/// unusual, GGUF encoding `MetadataValue` itself carries) must also be
/// read, not just the more common `F32` encoding -- proves the reader
/// does not assume one wire width.
#[test]
fn architecture_from_metadata_reads_rope_freq_base_stored_as_f64() {
    use proxima_gguf::value::MetadataValue as Value;

    let embed_bytes = vec![0u8; 8 * 3 * 4]; // [embedding=8, vocab=3] f32
    let model = GgufModel {
        version: 3,
        metadata: vec![
            (
                "general.architecture".to_string(),
                Value::String("llama".to_string()),
            ),
            ("llama.embedding_length".to_string(), Value::U32(8)),
            ("llama.feed_forward_length".to_string(), Value::U32(32)),
            ("llama.attention.head_count".to_string(), Value::U32(2)),
            ("llama.attention.head_count_kv".to_string(), Value::U32(1)),
            ("llama.block_count".to_string(), Value::U32(4)),
            ("llama.rope.dimension_count".to_string(), Value::U32(4)),
            ("llama.rope.freq_base".to_string(), Value::F64(1_000_000.0)),
        ],
        tensors: vec![TensorPayload {
            name: "token_embd.weight".to_string(),
            dims: dims(&[8, 3]),
            ggml_type: WireType::F32,
            data: &embed_bytes,
        }],
    };
    let file_bytes = write_complete(&model).expect("writes gguf with an f64 rope.freq_base");
    let parsed = proxima_gguf::parse_complete(&file_bytes)
        .expect("parses gguf with an f64 rope.freq_base");

    let architecture = architecture_from_metadata(&parsed)
        .expect("derive architecture from real metadata keys");
    assert_eq!(
        architecture.rope_freq_base, 1_000_000.0,
        "an f64-encoded key must still be read"
    );
}

#[test]
fn architecture_from_metadata_names_the_missing_key() {
    let model = GgufModel {
        version: 3,
        metadata: Vec::new(),
        tensors: Vec::new(),
    };
    let file_bytes = write_complete(&model).expect("writes empty gguf");
    let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses empty gguf");

    let outcome = architecture_from_metadata(&parsed);
    assert!(matches!(
        outcome,
        Err(InteropError::MissingMetadataKey { key }) if key == "general.architecture"
    ));
}

/// Builds a one-tensor GGUF fixture whose `blk.0.ffn_down.weight` is
/// `ggml_type` real on-disk bytes for `rows` rows of `k` elements each --
/// the shared setup every `weight_precision` recode test below starts
/// from, so each test's own body is the recode assertion, not fixture
/// plumbing.
// every caller below is a `#[cfg(feature = "std")]` recode test -- bare
// `cargo test` (no features) compiles this with no reachable caller and
// `-D dead-code` rejects it.
#[cfg(feature = "std")]
fn one_tensor_gguf(
    ggml_type: WireType,
    data: &[u8],
    rows: u64,
    k: u64,
) -> (Vec<u8>, alloc::string::String) {
    let name = "blk.0.ffn_down.weight".to_string();
    let model = GgufModel {
        version: 3,
        metadata: Vec::new(),
        tensors: vec![TensorPayload {
            name: name.clone(),
            dims: dims(&[k, rows]),
            ggml_type,
            data,
        }],
    };
    (
        write_complete(&model).expect("writes single-tensor gguf fixture"),
        name,
    )
}

/// (a): a `weight_precision` rule recoding one `Q4_K` tensor to `Q8_0`
/// lands in [`BoundWeights::packed_owned`] tagged [`Codec::Q8_0`],
/// `bytes_after` matches `Q8_0`'s own block arithmetic for the tensor's
/// dims, and dequantizing the recoded bytes agrees with dequantizing the
/// original `Q4_K` bytes within `Q8_0`'s own computed quantization error
/// bound (`d = amax / 127`, `proxima_gguf::quant::q8_0::quantize`'s own
/// doc) -- not `assert!(close)`, the actual per-block bound the source
/// formula produces.
#[cfg(feature = "std")]
#[test]
fn weight_precision_rule_recodes_q4_k_to_q8_0_within_q8_0_error_bound() {
    use proxima_gguf::quant::{q4_k, q8_0};

    let rows = 2usize;
    let k = q4_k::QK_K;
    let element_count = rows * k;
    let original_f32: Vec<f32> = (0..element_count)
        .map(|index| ((index % 37) as f32 - 18.0) * 0.37)
        .collect();

    let mut q4k_bytes = vec![0u8; rows * q4_k::BLOCK_BYTES];
    q4_k::quantize(&original_f32, &mut q4k_bytes)
        .expect("q4_k::quantize handles a multi-super-block run directly");

    let (file_bytes, name) = one_tensor_gguf(WireType::Q4_K, &q4k_bytes, rows as u64, k as u64);
    let parsed =
        proxima_gguf::pipe::parse_complete(&file_bytes).expect("parses q4_k gguf fixture");

    let rules = [crate::serving::WeightPrecisionRule {
        pattern: crate::serving::NamePattern::Exact(name.as_str()),
        target: GgmlType::Q8_0,
    }];
    let mut state = BoundWeights {
        resident_bytes: 0,
        owned: Vec::new(),
        packed: Vec::new(),
        packed_owned: Vec::new(),
        precision: &rules,
    };
    bind_dense_as(
        &parsed,
        &file_bytes,
        &name,
        "recoded".to_string(),
        &mut state,
    )
    .expect("recodes a q4_k tensor to q8_0");

    assert!(
        state.packed.is_empty(),
        "a recoded tensor must not also take the on-disk packed-codec path"
    );
    assert!(
        state.owned.is_empty(),
        "a q8_0 target must not land in the owned f32 slot"
    );
    assert_eq!(
        state.packed_owned.len(),
        1,
        "exactly one recoded tensor expected"
    );
    let (recoded_name, recoded_bytes, kind) = &state.packed_owned[0];
    assert_eq!(
        recoded_name, "recoded@q8_0",
        "a recode binds under a NEW proved name, never overwriting `target_name` itself"
    );
    assert_eq!(*kind, Codec::Q8_0);

    let expected_bytes = element_count / q8_0::QK8_0 * q8_0::BLOCK_BYTES;
    assert_eq!(recoded_bytes.len(), expected_bytes);
    assert_eq!(state.resident_bytes, expected_bytes);

    let mut original_dequant = vec![0.0f32; element_count];
    q4_k::dequantize(&q4k_bytes, &mut original_dequant)
        .expect("dequantizes the original q4_k bytes");
    let mut recoded_dequant = vec![0.0f32; element_count];
    q8_0::dequantize(recoded_bytes, &mut recoded_dequant)
        .expect("dequantizes the recoded q8_0 bytes");

    for (block_index, block) in original_dequant
        .as_chunks::<{ q8_0::QK8_0 }>()
        .0
        .iter()
        .enumerate()
    {
        let block_max = block.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        // `quantize_block`'s own formula: `d = amax / 127`, each level
        // rounds to the nearest multiple of `d`, so the base per-element
        // rounding error is half that step. The block scale itself is
        // stored as `f16` (half ULP relative error `2^-11`), and that
        // error is scaled by the largest representable level (127), so
        // the two terms combine as `d * (0.5 + 127 * 2^-11)`, not `d/2`
        // alone -- omitting the second term is what made this bound too
        // tight to hold for every block.
        let delta = block_max / 127.0;
        let scale_storage_error = 127.0 * delta * 2f32.powi(-11);
        let bound = delta * 0.5 + scale_storage_error + f32::EPSILON;
        let start = block_index * q8_0::QK8_0;
        for offset in 0..q8_0::QK8_0 {
            let difference =
                (original_dequant[start + offset] - recoded_dequant[start + offset]).abs();
            assert!(
                difference <= bound,
                "element {} differs by {difference}, exceeds q8_0's own half-step bound {bound}",
                start + offset
            );
        }
    }
}

/// (b): a `weight_precision` rule targeting `F32` lands the recoded
/// tensor in [`BoundWeights::owned`], bit-identical to dequantizing the
/// same on-disk bytes directly through [`gguf_tensor_as_f32`].
#[cfg(feature = "std")]
#[test]
fn weight_precision_rule_recodes_to_f32_bit_identical_to_direct_dequant() {
    use proxima_gguf::quant::q4_k;

    let rows = 2usize;
    let k = q4_k::QK_K;
    let element_count = rows * k;
    let original_f32: Vec<f32> = (0..element_count)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.11)
        .collect();
    let mut q4k_bytes = vec![0u8; rows * q4_k::BLOCK_BYTES];
    q4_k::quantize(&original_f32, &mut q4k_bytes)
        .expect("q4_k::quantize handles a multi-super-block run directly");

    let (file_bytes, name) = one_tensor_gguf(WireType::Q4_K, &q4k_bytes, rows as u64, k as u64);
    let parsed =
        proxima_gguf::pipe::parse_complete(&file_bytes).expect("parses q4_k gguf fixture");

    let direct_dequant =
        gguf_tensor_as_f32(&parsed, &file_bytes, &name).expect("direct dequant baseline");

    let rules = [crate::serving::WeightPrecisionRule {
        pattern: crate::serving::NamePattern::Exact(name.as_str()),
        target: GgmlType::F32,
    }];
    let mut state = BoundWeights {
        resident_bytes: 0,
        owned: Vec::new(),
        packed: Vec::new(),
        packed_owned: Vec::new(),
        precision: &rules,
    };
    bind_dense_as(
        &parsed,
        &file_bytes,
        &name,
        "recoded".to_string(),
        &mut state,
    )
    .expect("recodes a q4_k tensor to f32");

    assert!(state.packed.is_empty());
    assert!(state.packed_owned.is_empty());
    assert_eq!(state.owned.len(), 1);
    let (recoded_name, recoded_values) = &state.owned[0];
    assert_eq!(
        recoded_name, "recoded@f32",
        "a recode binds under a NEW proved name, never overwriting `target_name` itself"
    );
    assert_eq!(
        recoded_values, &direct_dequant,
        "an f32-target recode must be bit-identical to dequantizing the same bytes directly"
    );
    assert_eq!(
        state.resident_bytes,
        direct_dequant.len() * core::mem::size_of::<f32>()
    );
}

/// (c): a `weight_precision` rule naming a target with no
/// [`proxima_gguf::quant`] encoder (`Iq1S`) surfaces
/// [`InteropError::UnsupportedWeightPrecisionTarget`] rather than
/// silently falling back to the on-disk codec.
#[cfg(feature = "std")]
#[test]
fn weight_precision_rule_to_unencodable_target_names_the_typed_error() {
    use proxima_gguf::quant::q4_k;

    let rows = 1usize;
    let k = q4_k::QK_K;
    let element_count = rows * k;
    let original_f32 = vec![0.5f32; element_count];
    let mut q4k_bytes = vec![0u8; rows * q4_k::BLOCK_BYTES];
    q4_k::quantize(&original_f32, &mut q4k_bytes).expect("one QK_K-sized row");

    let (file_bytes, name) = one_tensor_gguf(WireType::Q4_K, &q4k_bytes, rows as u64, k as u64);
    let parsed =
        proxima_gguf::pipe::parse_complete(&file_bytes).expect("parses q4_k gguf fixture");

    let rules = [crate::serving::WeightPrecisionRule {
        pattern: crate::serving::NamePattern::Exact(name.as_str()),
        target: GgmlType::Iq1S,
    }];
    let mut state = BoundWeights {
        resident_bytes: 0,
        owned: Vec::new(),
        packed: Vec::new(),
        packed_owned: Vec::new(),
        precision: &rules,
    };
    let error = bind_dense_as(
        &parsed,
        &file_bytes,
        &name,
        "recoded".to_string(),
        &mut state,
    )
    .expect_err("iq1_s has no proxima_gguf::quant encoder");
    // `Codec` now recognizes `Iq1S` (`codec_from_ggml_type` is 1:1 over
    // every `GgmlType` a `Codec` names), so the rejection moves one level
    // deeper than `UnsupportedWeightPrecisionTarget` -- into
    // `quantize_to_kind`'s own `UnsupportedCodec` arm, exactly the shape
    // `codec_from_ggml_type`'s own doc already predicted for `Q5_1`/`Q5_0`.
    assert!(matches!(
        error,
        InteropError::Quant(QuantError::UnsupportedCodec { codec }) if codec == "iq1_s"
    ));
}

/// (d): no matching rule leaves [`bind_dense_as`]'s output byte-identical
/// to before `weight_precision` existed -- the no-op path is proven, not
/// assumed.
#[cfg(feature = "std")]
#[test]
fn no_matching_weight_precision_rule_is_byte_identical_to_the_no_rule_path() {
    use proxima_gguf::quant::q4_k;

    let rows = 1usize;
    let k = q4_k::QK_K;
    let element_count = rows * k;
    let original_f32: Vec<f32> = (0..element_count)
        .map(|index| index as f32 * 0.01)
        .collect();
    let mut q4k_bytes = vec![0u8; rows * q4_k::BLOCK_BYTES];
    q4_k::quantize(&original_f32, &mut q4k_bytes).expect("one QK_K-sized row");

    let (file_bytes, name) = one_tensor_gguf(WireType::Q4_K, &q4k_bytes, rows as u64, k as u64);
    let parsed =
        proxima_gguf::pipe::parse_complete(&file_bytes).expect("parses q4_k gguf fixture");

    // A rule that names a DIFFERENT tensor must never fire here.
    let non_matching_rules = [crate::serving::WeightPrecisionRule {
        pattern: crate::serving::NamePattern::Exact("blk.9.ffn_down.weight"),
        target: GgmlType::Q8_0,
    }];
    let mut with_rules = BoundWeights {
        resident_bytes: 0,
        owned: Vec::new(),
        packed: Vec::new(),
        packed_owned: Vec::new(),
        precision: &non_matching_rules,
    };
    bind_dense_as(
        &parsed,
        &file_bytes,
        &name,
        "target".to_string(),
        &mut with_rules,
    )
    .expect("binds with a non-matching rule set present");

    let mut without_rules = BoundWeights {
        resident_bytes: 0,
        owned: Vec::new(),
        packed: Vec::new(),
        packed_owned: Vec::new(),
        precision: &[],
    };
    bind_dense_as(
        &parsed,
        &file_bytes,
        &name,
        "target".to_string(),
        &mut without_rules,
    )
    .expect("binds with no rule set at all");

    assert_eq!(with_rules.resident_bytes, without_rules.resident_bytes);
    assert_eq!(with_rules.owned, without_rules.owned);
    assert_eq!(with_rules.packed.len(), without_rules.packed.len());
    assert_eq!(
        with_rules.packed_owned, without_rules.packed_owned,
        "a non-matching rule must leave bind_dense_as byte-identical to having no rules"
    );
    for (with_block, without_block) in with_rules.packed.iter().zip(without_rules.packed.iter())
    {
        assert_eq!(with_block.0, without_block.0);
    }
}

/// (I5): binding the SAME source tensor twice under the SAME
/// `target_name` but two DIFFERENT `weight_precision` targets in
/// sequence produces two DISTINCT proved names
/// (`{target_name}@q8_0`/`{target_name}@q5_k`), and the first call's own
/// entry is byte-for-byte untouched by the second -- a later policy
/// change is a rebind to a new name, never an in-place rewrite of the
/// name a plan already bound against ([`recode_tensor`]'s own doc).
#[cfg(feature = "std")]
#[test]
fn sequential_weight_precision_rebinds_produce_distinct_names_and_leave_the_first_entry_untouched()
 {
    use proxima_gguf::quant::q4_k;

    let rows = 1usize;
    let k = q4_k::QK_K;
    let element_count = rows * k;
    let original_f32: Vec<f32> = (0..element_count)
        .map(|index| index as f32 * 0.02)
        .collect();
    let mut q4k_bytes = vec![0u8; rows * q4_k::BLOCK_BYTES];
    q4_k::quantize(&original_f32, &mut q4k_bytes).expect("one QK_K-sized row");

    let (file_bytes, name) = one_tensor_gguf(WireType::Q4_K, &q4k_bytes, rows as u64, k as u64);
    let parsed =
        proxima_gguf::pipe::parse_complete(&file_bytes).expect("parses q4_k gguf fixture");

    let first_rules = [crate::serving::WeightPrecisionRule {
        pattern: crate::serving::NamePattern::Exact(name.as_str()),
        target: GgmlType::Q8_0,
    }];
    let second_rules = [crate::serving::WeightPrecisionRule {
        pattern: crate::serving::NamePattern::Exact(name.as_str()),
        target: GgmlType::Q5_K,
    }];

    let mut state = BoundWeights {
        resident_bytes: 0,
        owned: Vec::new(),
        packed: Vec::new(),
        packed_owned: Vec::new(),
        precision: &first_rules,
    };
    bind_dense_as(
        &parsed,
        &file_bytes,
        &name,
        "shared_target".to_string(),
        &mut state,
    )
    .expect("first recode, to q8_0");

    let entry_after_first = state.packed_owned[0].clone();

    state.precision = &second_rules;
    bind_dense_as(
        &parsed,
        &file_bytes,
        &name,
        "shared_target".to_string(),
        &mut state,
    )
    .expect("second recode, to q5_k, under the same target_name base");

    assert_eq!(
        state.packed_owned.len(),
        2,
        "a rebind must APPEND a new entry, never replace the first"
    );
    assert_eq!(
        state.packed_owned[0], entry_after_first,
        "the first call's own entry must be byte-for-byte untouched by the second call"
    );
    assert_eq!(state.packed_owned[0].0, "shared_target@q8_0");
    assert_eq!(state.packed_owned[1].0, "shared_target@q5_k");
    assert_ne!(
        state.packed_owned[0].0, state.packed_owned[1].0,
        "two different precision targets for the same target_name must produce distinct names"
    );
}

/// (e): [`ServingConfig`]'s config-as-mirror -- a config built as a full
/// struct literal with an explicit `weight_precision` slice and one
/// built via [`ServingConfig::with_weight_precision`]'s fluent builder
/// agree bit for bit, the same interoperability
/// [`crate::serving::tests::kv_bucket_tokens_agrees_across_literal_and_default_override`]
/// already proves for a plain field.
#[test]
fn weight_precision_config_and_builder_surfaces_agree() {
    use crate::serving::{NamePattern, ServingConfig, WeightPrecisionRule};

    let rules = [WeightPrecisionRule {
        pattern: NamePattern::Suffix("_exps.weight"),
        target: GgmlType::Q8_0,
    }];

    let via_builder = ServingConfig::default().with_weight_precision(&rules);
    let via_field = ServingConfig {
        weight_precision: &rules,
        ..ServingConfig::default()
    };

    assert_eq!(via_builder, via_field);
    assert_eq!(via_builder.weight_precision, &rules);
}
