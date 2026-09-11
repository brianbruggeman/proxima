//! A quantized `token_embd.weight` used ONLY as an embedding gather (never as
//! a matmul operand) used to be rejected by `reject_non_float32`'s f32-only
//! validation gate before `run_node_into`'s own `quantized_gather_operand`
//! dispatch ever got a chance to run it -- `is_quantized_matmul_operand`
//! (`src/cpu.rs`) only ever recognized the multiply-then-reduce matmul shape,
//! never the identity-gather shape `embedding_lookup` builds. Compiled
//! against `proxima_tensor`'s public API only (integration test, not
//! `#[cfg(test)] mod tests`), so this is the one artifact that can actually
//! settle whether a foreign caller's quantized embedding table lowers.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::spec::{embedding_lookup, input_leaf};
use proxima_tensor::{DType, Extent};

fn filled(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        })
        .collect()
}

/// Runs a gather-only quantized embedding table through the real production
/// entry point ([`proxima_tensor::cpu::evaluate_quantized_named`], the same
/// function `evaluate_quantized_with_scratch`'s callers use) and checks it
/// bit-for-bit against an independently computed dequantize-then-index f32
/// reference -- guiding-principle 14: the dequantize-then-lookup reference is
/// correct by construction, not this fix checked against itself.
fn assert_quantized_gather_matches_reference(
    embedding: u32,
    table_block: QuantizedBlock<'_>,
    table_f32: &[f32],
) {
    let vocab = 3u32;
    let tokens = 2usize;

    let mut program = Vec::new();
    let ids = input_leaf(&mut program, DType::Int32, vec![Extent::Symbolic(0)], "ids");
    let table = input_leaf(
        &mut program,
        DType::UInt8,
        vec![Extent::Static(vocab), Extent::Static(embedding)],
        "token_embd.weight",
    );
    let embedded = embedding_lookup(&mut program, table, ids);

    let ids_data: Vec<f32> = vec![2.0, 0.0];
    let named: Vec<(&str, QuantizedBlock)> = vec![
        ("ids", QuantizedBlock::Float32(ids_data.as_slice())),
        ("token_embd.weight", table_block),
    ];

    let evaluated = proxima_tensor::cpu::evaluate_quantized_named(
        &program,
        &[tokens as u64],
        &named,
        &[embedded],
    )
    .expect("a quantized table used only as an embedding gather lowers");
    let (values, shape) = evaluated
        .get(embedded)
        .expect("embedding gather output present");
    assert_eq!(shape, [tokens as u64, embedding as u64]);

    for (token, row) in ids_data.iter().zip(values.chunks_exact(embedding as usize)) {
        let vocab_row = *token as usize;
        let expected =
            &table_f32[vocab_row * embedding as usize..(vocab_row + 1) * embedding as usize];
        for (actual, expected) in row.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 5e-2,
                "dequantized gather row diverges from the reference table row: actual={actual} expected={expected}"
            );
        }
    }
}

#[test]
fn quantized_embedding_gather_matches_dequantized_reference_q8_0() {
    use proxima_gguf::quant::q8_0::{BLOCK_BYTES, QK8_0, quantize};

    let embedding = 32u32;
    let table_f32 = filled(0x2026_0908_dead_beef, 3 * embedding as usize);
    let mut table_bytes = vec![0u8; (table_f32.len() / QK8_0) * BLOCK_BYTES];
    quantize(&table_f32, &mut table_bytes).expect("quantize fixture to q8_0");

    assert_quantized_gather_matches_reference(
        embedding,
        QuantizedBlock::Q8_0(&table_bytes),
        &table_f32,
    );
}

#[test]
fn quantized_embedding_gather_matches_dequantized_reference_q4_k() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    let embedding = 256u32;
    let table_f32 = filled(0x2026_0908_dead_beef, 3 * embedding as usize);
    let mut table_bytes = vec![0u8; (table_f32.len() / QK_K) * BLOCK_BYTES];
    quantize(&table_f32, &mut table_bytes).expect("quantize fixture to q4_k");

    assert_quantized_gather_matches_reference(
        embedding,
        QuantizedBlock::Q4K(&table_bytes),
        &table_f32,
    );
}
