//! `PROXIMA_MULTI_ROW_UNROLL=1` A/B, codec-agnostic:
//! the current generic multi-row arm's `q`/`s` loops
//! (`push_packed_row_multi_row_unroll_body`, `omega/src/msl/
//! elementwise_reduce_core.rs`) render fully unrolled with literal indices
//! at all three sites that touch `sumf` (zero-init, accumulate, epilogue
//! reduce/write) instead of a dynamic `for` loop -- same decode call
//! (`operand_read(weight, index, Some(codec))`), same `k` order, same
//! `q`-outer/`s`-inner nesting, same `push_body_steps`/`combine_fn`
//! expression text, only the induction variables become literals. This gate
//! proves the unroll changes no arithmetic: for every `(codec, tokens, K,
//! rows)` shape, `PROXIMA_MULTI_ROW_UNROLL=1` must produce BIT-EXACT
//! (`to_bits()`) output against the unset-env default on the SAME packed
//! weight/activation bytes.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};

#[derive(Clone, Copy)]
enum TestCodec {
    Q4_0,
    Q8_0,
    Q4K,
    Q6K,
}

impl TestCodec {
    fn codec(self) -> omega::Codec {
        match self {
            TestCodec::Q4_0 => omega::Codec::Q4_0,
            TestCodec::Q8_0 => omega::Codec::Q8_0,
            TestCodec::Q4K => omega::Codec::Q4K,
            TestCodec::Q6K => omega::Codec::Q6K,
        }
    }

    fn block_elements(self) -> usize {
        match self {
            TestCodec::Q4_0 => proxima_gguf::quant::q4_0::QK4_0,
            TestCodec::Q8_0 => proxima_gguf::quant::q8_0::QK8_0,
            TestCodec::Q4K | TestCodec::Q6K => proxima_gguf::quant::q4_k::QK_K,
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            TestCodec::Q4_0 => proxima_gguf::quant::q4_0::BLOCK_BYTES,
            TestCodec::Q8_0 => proxima_gguf::quant::q8_0::BLOCK_BYTES,
            TestCodec::Q4K => proxima_gguf::quant::q4_k::BLOCK_BYTES,
            TestCodec::Q6K => proxima_gguf::quant::q6_k::BLOCK_BYTES,
        }
    }

    fn quantize(self, input: &[f32], output: &mut [u8]) {
        match self {
            TestCodec::Q4_0 => proxima_gguf::quant::q4_0::quantize(input, output).unwrap(),
            TestCodec::Q8_0 => proxima_gguf::quant::q8_0::quantize(input, output).unwrap(),
            TestCodec::Q4K => proxima_gguf::quant::q4_k::quantize(input, output).unwrap(),
            TestCodec::Q6K => proxima_gguf::quant::q6_k::quantize(input, output).unwrap(),
        }
    }
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// `[out_dim, in_dim] x [tokens, in_dim] -> [tokens, out_dim]`, reduced over
/// `in_dim` -- each operand stores `in_dim` as its own innermost, contiguous
/// dim, matching this file's sibling A/B test binaries' own `matmul_program`
/// construction.
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

fn pack_rows(codec: TestCodec, rows: &[Vec<f32>], in_dim: usize) -> Vec<u8> {
    let blocks_per_row = in_dim / codec.block_elements();
    let mut packed = vec![0u8; rows.len() * blocks_per_row * codec.block_bytes()];
    for (row, row_packed) in rows
        .iter()
        .zip(packed.chunks_exact_mut(blocks_per_row * codec.block_bytes()))
    {
        codec.quantize(row, row_packed);
    }
    packed
}

/// Runs the multi-token matmul once under the given
/// `PROXIMA_MULTI_ROW_UNROLL` env value (`None` = unset, today's
/// default emit) and returns the raw output bit patterns -- `to_bits()`, not
/// the floats, for exact-equality comparison.
fn run_bits(
    codec: TestCodec,
    tokens: usize,
    in_dim: usize,
    out_dim: usize,
    seed: u64,
    unroll_env: Option<&str>,
) -> Vec<u32> {
    let rows: Vec<Vec<f32>> = (0..out_dim)
        .map(|row| random_vec(seed + row as u64, in_dim))
        .collect();
    let packed = pack_rows(codec, &rows, in_dim);
    let activation = random_vec(seed + 9973, tokens * in_dim);
    let (program, sum) = matmul_program(tokens as u32, in_dim as u32, out_dim as u32);
    let blocks = [
        QuantizedBlock::Packed {
            codec: codec.codec(),
            bytes: &packed,
        },
        QuantizedBlock::Float32(&activation),
    ];

    let output = temp_env::with_var("PROXIMA_MULTI_ROW_UNROLL", unroll_env, || {
        let plan = omega::plan(&program, &[], &blocks, &[sum], NumericPolicy::default())
            .expect("metal plans the multi-token packed matmul");
        omega::execute_plan(&plan, &blocks).expect("metal runs the matmul on a real device")
    });
    output.root().iter().map(|value| value.to_bits()).collect()
}

fn plan_kernel_keys(
    codec: TestCodec,
    tokens: usize,
    in_dim: usize,
    out_dim: usize,
    seed: u64,
    unroll_env: Option<&str>,
) -> Vec<String> {
    let rows: Vec<Vec<f32>> = (0..out_dim)
        .map(|row| random_vec(seed + row as u64, in_dim))
        .collect();
    let packed = pack_rows(codec, &rows, in_dim);
    let activation = random_vec(seed + 9973, tokens * in_dim);
    let (program, sum) = matmul_program(tokens as u32, in_dim as u32, out_dim as u32);
    let blocks = [
        QuantizedBlock::Packed {
            codec: codec.codec(),
            bytes: &packed,
        },
        QuantizedBlock::Float32(&activation),
    ];
    temp_env::with_var("PROXIMA_MULTI_ROW_UNROLL", unroll_env, || {
        let plan = omega::plan(&program, &[], &blocks, &[sum], NumericPolicy::default())
            .expect("metal plans the multi-token packed matmul");
        plan.kernel_keys().expect("plan reports its own kernel cache keys")
    })
}

/// A codec whose current generic-arm multi-row body is a fast codec body
/// (`fast_q4k_active`, not the generic arm this unroll experiment
/// reproduces) must NEVER be admitted -- `multi_row_generic_arm_current`'s
/// own doc names this residual, shared with tg-share/register-tile. Proves non-admission
/// three ways: the cache key never carries `_u`, the cache key is
/// IDENTICAL whether the override is set, and (trivially, since the same
/// kernel renders either way) the output bits are identical too.
fn assert_unroll_not_admitted(codec: TestCodec, tokens: usize, k: usize, rows: usize, seed: u64) {
    let baseline_keys = plan_kernel_keys(codec, tokens, k, rows, seed, None);
    let shared_keys = plan_kernel_keys(codec, tokens, k, rows, seed, Some("1"));
    assert!(
        !shared_keys.iter().any(|key| key.contains("_u")),
        "a fast-body codec's cache key must never carry _u even with the override set: {shared_keys:?}"
    );
    assert_eq!(
        baseline_keys, shared_keys,
        "a fast-body codec's cache key must be identical whether PROXIMA_MULTI_ROW_UNROLL is set"
    );
    assert_unroll_bit_exact(codec, tokens, k, rows, seed);
}

/// One `(codec, tokens, K, rows)` case: `PROXIMA_MULTI_ROW_UNROLL=1`
/// must reproduce the unset-env default bit-for-bit.
fn assert_unroll_bit_exact(codec: TestCodec, tokens: usize, k: usize, rows: usize, seed: u64) {
    let baseline = run_bits(codec, tokens, k, rows, seed, None);
    let shared = run_bits(codec, tokens, k, rows, seed, Some("1"));
    assert_eq!(
        baseline.len(),
        tokens * rows,
        "degenerate gate: tokens={tokens} K={k} rows={rows} baseline produced the wrong element count"
    );
    assert_eq!(
        baseline, shared,
        "tokens={tokens} K={k} rows={rows}: PROXIMA_MULTI_ROW_UNROLL=1 produced different \
         bits than the unset-env default -- the register-tile staging must not reorder or alter \
         any output's accumulation"
    );
}

macro_rules! unroll_ab_case {
    ($name:ident, $codec:expr, $tokens:expr, $k:expr, $rows:expr, $seed:expr) => {
        #[test]
        fn $name() {
            assert_unroll_bit_exact($codec, $tokens, $k, $rows, $seed);
        }
    };
}

/// Fast-body codecs (`Q4_K`'s `fast_q4k_active` pair-dot body at M>1) must
/// assert NON-admission, not bit-exact equality across the override --
/// [`assert_unroll_not_admitted`]'s own doc.
macro_rules! unroll_ab_not_admitted_case {
    ($name:ident, $codec:expr, $tokens:expr, $k:expr, $rows:expr, $seed:expr) => {
        #[test]
        fn $name() {
            assert_unroll_not_admitted($codec, $tokens, $k, $rows, $seed);
        }
    };
}

unroll_ab_case!(q4_0_tokens27_k1536_rows12288, TestCodec::Q4_0, 27, 1536, 12288, 511);
unroll_ab_case!(q4_0_tokens600_k1536_rows12288, TestCodec::Q4_0, 600, 1536, 12288, 521);
unroll_ab_case!(q4_0_tokens600_k12288_rows1536, TestCodec::Q4_0, 600, 12288, 1536, 523);
unroll_ab_case!(q4_0_tokens9_k1536_rows6144_tail, TestCodec::Q4_0, 9, 1536, 6144, 541);
unroll_ab_case!(q4_0_tokens1_k1536_rows6144_single_token, TestCodec::Q4_0, 1, 1536, 6144, 547);

unroll_ab_case!(q8_0_tokens27_k1536_rows12288, TestCodec::Q8_0, 27, 1536, 12288, 557);
unroll_ab_case!(q8_0_tokens600_k1536_rows12288, TestCodec::Q8_0, 600, 1536, 12288, 563);
unroll_ab_case!(q8_0_tokens600_k12288_rows1536, TestCodec::Q8_0, 600, 12288, 1536, 569);
unroll_ab_case!(q8_0_tokens9_k1536_rows6144_tail, TestCodec::Q8_0, 9, 1536, 6144, 571);
unroll_ab_case!(q8_0_tokens1_k1536_rows6144_single_token, TestCodec::Q8_0, 1, 1536, 6144, 577);

unroll_ab_not_admitted_case!(q4k_tokens27_k1536_rows12288_not_admitted, TestCodec::Q4K, 27, 1536, 12288, 587);
unroll_ab_not_admitted_case!(q4k_tokens600_k1536_rows12288_not_admitted, TestCodec::Q4K, 600, 1536, 12288, 593);
unroll_ab_not_admitted_case!(q4k_tokens9_k1536_rows6144_tail_not_admitted, TestCodec::Q4K, 9, 1536, 6144, 599);
unroll_ab_case!(q4k_tokens1_k1536_rows6144_single_token, TestCodec::Q4K, 1, 1536, 6144, 601);

unroll_ab_case!(q6k_tokens27_k1536_rows12288, TestCodec::Q6K, 27, 1536, 12288, 607);
unroll_ab_case!(q6k_tokens600_k1536_rows12288, TestCodec::Q6K, 600, 1536, 12288, 613);
unroll_ab_case!(q6k_tokens600_k12288_rows1536, TestCodec::Q6K, 600, 12288, 1536, 617);
unroll_ab_case!(q6k_tokens9_k1536_rows6144_tail, TestCodec::Q6K, 9, 1536, 6144, 619);
unroll_ab_case!(q6k_tokens1_k1536_rows6144_single_token, TestCodec::Q6K, 1, 1536, 6144, 631);
