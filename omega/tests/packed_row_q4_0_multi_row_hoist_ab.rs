//! `PROXIMA_Q4_0_MULTI_ROW_HOIST=1` A/B (owner brief, prefill correction
//! slice): `push_packed_row_multi_row_body`'s `Codec::Q4_0` fast arm
//! (`push_packed_row_multi_row_q4_0_body`, `omega/src/msl/
//! elementwise_reduce_core.rs`) hoists the per-block `d` header decode
//! once per 32-element block per feature via `simd_broadcast_first`,
//! instead of every one of the 32 lanes sharing that block re-deriving `d`
//! independently. This gate proves the hoist changes no arithmetic: for
//! every `(tokens, K, rows)` shape below, `PROXIMA_Q4_0_MULTI_ROW_HOIST=1`
//! must produce BIT-EXACT (`to_bits()`) output against the unset-env
//! default on the SAME real-Q4_0-shaped weights/activations -- the hoisted
//! `d`/nibble values are byte-identical to what `q4_0_element` computes
//! today, so any bit divergence means the hoist reordered or altered the
//! per-output accumulation, not merely lost precision.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// `[out_dim, in_dim] x [tokens, in_dim] -> [tokens, out_dim]`, reduced over
/// `in_dim` -- the same multi-token matmul shape
/// `packed_row_multi_activation_parity.rs` builds for the K-quant codecs,
/// here packed `Q4_0` instead so `push_packed_row_multi_row_body`'s
/// `Codec::Q4_0` fast arm is the one under test.
fn matmul_program(tokens: u32, in_dim: u32, out_dim: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![Extent::Static(out_dim), Extent::Static(in_dim)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(tokens), Extent::Static(in_dim)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                (activation, IndexMap::Affine(projection(3, &[0, 2]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

fn pack_rows(rows: &[Vec<f32>], in_dim: usize) -> Vec<u8> {
    let blocks_per_row = in_dim / proxima_gguf::quant::q4_0::QK4_0;
    let mut packed =
        vec![0u8; rows.len() * blocks_per_row * proxima_gguf::quant::q4_0::BLOCK_BYTES];
    for (row, row_packed) in rows
        .iter()
        .zip(packed.chunks_exact_mut(blocks_per_row * proxima_gguf::quant::q4_0::BLOCK_BYTES))
    {
        proxima_gguf::quant::q4_0::quantize(row, row_packed).unwrap();
    }
    packed
}

/// Runs the multi-token matmul once under the given
/// `PROXIMA_Q4_0_MULTI_ROW_HOIST` env value (`None` = unset, today's
/// default emit) and returns the raw output bit patterns -- `to_bits()`,
/// not the floats, so a caller compares for exact equality rather than a
/// tolerance band. Deterministic seeded `Q4_0` bytes and activations,
/// sign-mixed (`Lcg::next_unit` spans `[-1, 1)`, no denormal-forcing) --
/// the same real-shaped-data posture `packed_row_rows_per_simdgroup_ab.rs`
/// already uses for this codec.
fn run_bits(
    tokens: usize,
    in_dim: usize,
    out_dim: usize,
    seed: u64,
    hoist_env: Option<&str>,
) -> Vec<u32> {
    let rows: Vec<Vec<f32>> = (0..out_dim)
        .map(|row| random_vec(seed + row as u64, in_dim))
        .collect();
    let packed = pack_rows(&rows, in_dim);
    let activation = random_vec(seed + 9973, tokens * in_dim);
    let (program, sum) = matmul_program(tokens as u32, in_dim as u32, out_dim as u32);
    let blocks = [
        QuantizedBlock::Packed {
            codec: omega::Codec::Q4_0,
            bytes: &packed,
        },
        QuantizedBlock::Float32(&activation),
    ];

    let output = temp_env::with_var("PROXIMA_Q4_0_MULTI_ROW_HOIST", hoist_env, || {
        let plan = omega::plan(&program, &[], &blocks, &[sum], NumericPolicy::default())
            .expect("metal plans the multi-token Q4_0 matmul");
        omega::execute_plan(&plan, &blocks).expect("metal runs the matmul on a real device")
    });
    output.root().iter().map(|value| value.to_bits()).collect()
}

/// One `(tokens, K, rows)` case: `PROXIMA_Q4_0_MULTI_ROW_HOIST=1` must
/// reproduce the unset-env default bit-for-bit.
fn assert_hoist_bit_exact(tokens: usize, k: usize, rows: usize, seed: u64) {
    let baseline = run_bits(tokens, k, rows, seed, None);
    let hoisted = run_bits(tokens, k, rows, seed, Some("1"));
    assert_eq!(
        baseline.len(),
        tokens * rows,
        "degenerate gate: tokens={tokens} K={k} rows={rows} baseline produced the wrong element count"
    );
    assert_eq!(
        baseline, hoisted,
        "tokens={tokens} K={k} rows={rows}: PROXIMA_Q4_0_MULTI_ROW_HOIST=1 produced different \
         bits than the unset-env default -- the header-decode hoist must not reorder or alter \
         any output's accumulation"
    );
}

#[test]
fn tokens27_k1536_rows12288_bit_exact_across_hoist() {
    assert_hoist_bit_exact(27, 1536, 12288, 11);
}

#[test]
fn tokens27_k1536_rows6144_bit_exact_across_hoist() {
    assert_hoist_bit_exact(27, 1536, 6144, 23);
}

#[test]
fn tokens27_k12288_rows1536_bit_exact_across_hoist() {
    assert_hoist_bit_exact(27, 12288, 1536, 37);
}

#[test]
fn tokens27_k1536_rows2048_bit_exact_across_hoist() {
    assert_hoist_bit_exact(27, 1536, 2048, 41);
}

#[test]
fn tokens8_k1536_rows6144_bit_exact_across_hoist() {
    assert_hoist_bit_exact(8, 1536, 6144, 53);
}

/// `9` crosses the default `PACKED_ROW_ACTIVATION_GROUP=8` tile boundary --
/// a partial second token-group -- forcing the fast arm's `s`-loop
/// bounds-check (`token_flat < token_total`) to actually gate a write.
#[test]
fn tokens9_k1536_rows6144_bit_exact_across_hoist_partial_group() {
    assert_hoist_bit_exact(9, 1536, 6144, 59);
}

/// A single token must still route through the SAME `push_packed_row_
/// multi_row_body` (`token_total > 1` is the gate on the caller side, not
/// on the fast-arm itself; `push_packed_row_blocked_body` only calls into
/// the multi-row body at all when `packed_row_block_token_total > 1`, so
/// this token count exercises the OTHER, single-row body, unaffected by
/// this change) -- named here to prove the two paths do not collide.
#[test]
fn tokens1_k1536_rows6144_routes_unaffected_by_hoist() {
    assert_hoist_bit_exact(1, 1536, 6144, 61);
}

/// Long bucket: several full `PACKED_ROW_ACTIVATION_GROUP=8` token groups
/// plus a tail, the shape closest to the captured prefill node 5063
/// (27 tokens) scaled up.
#[test]
fn tokens510_k1536_rows6144_bit_exact_across_hoist_long_bucket() {
    assert_hoist_bit_exact(510, 1536, 6144, 67);
}
