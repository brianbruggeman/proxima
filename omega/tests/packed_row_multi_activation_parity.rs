//! packed-row multi-row fold parity: the row-blocked packed matvec
//! (`msl.rs`'s `push_packed_row_multi_row_body`, folded into
//! `push_packed_row_blocked_body`'s `token_total > 1` branch) folds `s` activation
//! ("token") rows per streamed weight row instead of re-streaming the whole
//! weight once per row -- see that function's own doc for the mechanism.
//! This gate proves the fold is numerically transparent: for every codec
//! [`packed_row_block`] admits (Q4_K/Q5_K/Q6_K) and `s` in
//! `{1, 2, 4, 8, 9}` (9 crosses the default `group=8` tile boundary, forcing
//! two token-groups per feature group), the emitted kernel's output matches
//! an INDEPENDENT dequantize+dot reference -- not just CPU against Metal,
//! the same discipline `attn_multi_axis_tiled_gemm_parity.rs` already
//! applies to the tiled-GEMM path.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::cpu::evaluate_quantized;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};

#[derive(Clone, Copy)]
enum Codec {
    Q4K,
    Q5K,
    Q6K,
}

impl Codec {
    fn packed(self) -> omega::PackedCodec {
        match self {
            Codec::Q4K => omega::PackedCodec::Q4K,
            Codec::Q5K => omega::PackedCodec::Q5K,
            Codec::Q6K => omega::PackedCodec::Q6K,
        }
    }

    fn quantized_block(self, packed: &[u8]) -> QuantizedBlock<'_> {
        match self {
            Codec::Q4K => QuantizedBlock::Q4K(packed),
            Codec::Q5K => QuantizedBlock::Q5K(packed),
            Codec::Q6K => QuantizedBlock::Q6K(packed),
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            Codec::Q4K => proxima_gguf::quant::q4_k::BLOCK_BYTES,
            Codec::Q5K => proxima_gguf::quant::q5_k::BLOCK_BYTES,
            Codec::Q6K => proxima_gguf::quant::q6_k::BLOCK_BYTES,
        }
    }

    fn quantize(self, input: &[f32], output: &mut [u8]) {
        match self {
            Codec::Q4K => proxima_gguf::quant::q4_k::quantize(input, output).unwrap(),
            Codec::Q5K => proxima_gguf::quant::q5_k::quantize(input, output).unwrap(),
            Codec::Q6K => proxima_gguf::quant::q6_k::quantize(input, output).unwrap(),
        }
    }

    fn dequantize(self, data: &[u8], output: &mut [f32]) {
        match self {
            Codec::Q4K => proxima_gguf::quant::q4_k::dequantize(data, output).unwrap(),
            Codec::Q5K => proxima_gguf::quant::q5_k::dequantize(data, output).unwrap(),
            Codec::Q6K => proxima_gguf::quant::q6_k::dequantize(data, output).unwrap(),
        }
    }
}

const QK_K: usize = 256;

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// `[out_dim, in_dim] x [tokens, in_dim] -> [tokens, out_dim]`, reduced over
/// `in_dim` -- a plain matvec/matmul, one feature axis (weight-owned) and
/// one token axis (activation-owned), the shape
/// `classify_packed_row_multi_activation`'s axis-ownership split expects.
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
    // iteration space (tok, out, in): weight reads (out, in), ignoring tok;
    // activation reads (tok, in), ignoring out.
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

fn pack_rows(codec: Codec, rows: &[Vec<f32>], in_dim: usize) -> Vec<u8> {
    let blocks_per_row = in_dim / QK_K;
    let mut packed = vec![0u8; rows.len() * blocks_per_row * codec.block_bytes()];
    for (row, row_packed) in rows
        .iter()
        .zip(packed.chunks_exact_mut(blocks_per_row * codec.block_bytes()))
    {
        codec.quantize(row, row_packed);
    }
    packed
}

/// Dequantizes `packed` row by row and dots each row against every token's
/// activation row, computed independently of both the CPU quantized-matmul
/// path and omega's Metal emitter.
fn expected_output(
    codec: Codec,
    packed: &[u8],
    in_dim: usize,
    out_rows: usize,
    tokens: usize,
    activation: &[f32],
) -> Vec<f32> {
    let blocks_per_row = in_dim / QK_K;
    let mut dequantized_rows: Vec<Vec<f32>> = Vec::with_capacity(out_rows);
    for row_packed in packed.chunks_exact(blocks_per_row * codec.block_bytes()) {
        let mut row = vec![0.0f32; in_dim];
        codec.dequantize(row_packed, &mut row);
        dequantized_rows.push(row);
    }
    assert_eq!(dequantized_rows.len(), out_rows);

    let mut expected = vec![0.0f32; tokens * out_rows];
    for (token_index, token_activation) in activation.chunks_exact(in_dim).enumerate() {
        for (row_index, row) in dequantized_rows.iter().enumerate() {
            let dot: f32 = row
                .iter()
                .zip(token_activation.iter())
                .map(|(weight, value)| weight * value)
                .sum();
            expected[token_index * out_rows + row_index] = dot;
        }
    }
    expected
}

/// One `(codec, s)` case of the parity sweep, factored so the `#[test]`
/// functions below carry the semantically meaningful name (see the module
/// doc's own cross-product) instead of an index.
fn run_case(codec: Codec, tokens: usize) {
    const IN_DIM: usize = 512;
    const OUT_ROWS: usize = 24;
    // relative + absolute, same combined-bound shape
    // `attn_multi_axis_tiled_gemm_parity.rs` uses for the same reason: some
    // dot products land near a zero crossing, where a pure-relative bound is
    // unsound.
    const RELATIVE: f32 = 0.02;
    const ABSOLUTE: f32 = 0.1;

    let rows: Vec<Vec<f32>> = (0..OUT_ROWS)
        .map(|row| random_vec(61 + row as u64, IN_DIM))
        .collect();
    let packed = pack_rows(codec, &rows, IN_DIM);
    let activation = random_vec(97, tokens * IN_DIM);
    let expected = expected_output(codec, &packed, IN_DIM, OUT_ROWS, tokens, &activation);

    let (program, sum) = matmul_program(tokens as u32, IN_DIM as u32, OUT_ROWS as u32);
    let blocks = [codec.quantized_block(&packed), QuantizedBlock::Float32(&activation)];

    let shapes = proxima_tensor::infer(&program, &[]).expect("the synthetic program infers");
    let packed_operands: omega::PackedOperands =
        [(NodeId(0), codec.packed())].into_iter().collect();
    let mut bound =
        proxima_tensor::bind(&program, &shapes, &[sum], NumericPolicy::default()).expect("the synthetic program binds");
    proxima_tensor::correct_packed_matmul_layouts(&mut bound, &[NodeId(0)].into_iter().collect());
    let resolved = bound
        .iter()
        .find(|op| op.node == sum)
        .expect("the reduce node is bound");
    let kernel = omega::emit(resolved, &packed_operands, proxima_tensor::NumericPolicy::default()).expect("the synthetic program emits");
    // `metal-tiled-gemm` (`crate::sized::TILED_GEMM_MIN_TOKENS`, default 8)
    // is checked FIRST in `push_cooperative_reduce_body`'s own arm order and
    // legitimately supersedes this path once `tokens` clears that
    // threshold -- not invented to compose, the same posture every other
    // packed-row feature in this file's `Cargo.toml` doc takes. This gate's
    // job is the fold itself, not which of two independently-parity-tested
    // GEMM lanes a build with every feature on happens to prefer, so the
    // marker assertion only binds where this path is the only one that can
    // engage.
    if tokens > 1 && !cfg!(feature = "metal-tiled-gemm") {
        assert!(
            kernel.source.contains("token_group"),
            "s={tokens} > 1 must take the multi-activation fold, not the plain row-blocked path \
             (kernel source did not contain that path's own `token_group` marker)"
        );
    }

    let cpu = evaluate_quantized(&program, &[], &blocks, &[sum]).expect("cpu runs the matmul");
    let plan = omega::plan(&program, &[], &blocks, &[sum], NumericPolicy::default()).expect("metal plans the matmul");
    let metal =
        omega::execute_plan(&plan, &blocks).expect("metal runs the matmul on a real device");

    let element_count = tokens * OUT_ROWS;
    assert_eq!(cpu.root().len(), element_count, "degenerate gate: cpu produced no output");
    assert_eq!(metal.root().len(), element_count, "degenerate gate: metal produced no output");

    for (index, (&metal_value, &reference)) in metal.root().iter().zip(expected.iter()).enumerate()
    {
        let absolute = (metal_value - reference).abs();
        let bound = ABSOLUTE + RELATIVE * reference.abs();
        assert!(
            absolute <= bound,
            "s={tokens} element {index}: metal={metal_value} disagrees with the independent \
             dequantize+dot reference={reference} (abs_diff={absolute}, bound={bound})"
        );
    }
}

macro_rules! parity_case {
    ($name:ident, $codec:expr, $tokens:expr) => {
        #[test]
        fn $name() {
            run_case($codec, $tokens);
        }
    };
}

parity_case!(q4k_single_activation_row_matches_independent_reference, Codec::Q4K, 1);
parity_case!(q4k_two_activation_rows_match_independent_reference, Codec::Q4K, 2);
parity_case!(q4k_four_activation_rows_match_independent_reference, Codec::Q4K, 4);
parity_case!(q4k_eight_activation_rows_match_independent_reference, Codec::Q4K, 8);
parity_case!(
    q4k_nine_activation_rows_crosses_the_tile_boundary_and_matches_reference,
    Codec::Q4K,
    9
);

parity_case!(q5k_single_activation_row_matches_independent_reference, Codec::Q5K, 1);
parity_case!(q5k_two_activation_rows_match_independent_reference, Codec::Q5K, 2);
parity_case!(q5k_four_activation_rows_match_independent_reference, Codec::Q5K, 4);
parity_case!(q5k_eight_activation_rows_match_independent_reference, Codec::Q5K, 8);
parity_case!(
    q5k_nine_activation_rows_crosses_the_tile_boundary_and_matches_reference,
    Codec::Q5K,
    9
);

parity_case!(q6k_single_activation_row_matches_independent_reference, Codec::Q6K, 1);
parity_case!(q6k_two_activation_rows_match_independent_reference, Codec::Q6K, 2);
parity_case!(q6k_four_activation_rows_match_independent_reference, Codec::Q6K, 4);
parity_case!(q6k_eight_activation_rows_match_independent_reference, Codec::Q6K, 8);
parity_case!(
    q6k_nine_activation_rows_crosses_the_tile_boundary_and_matches_reference,
    Codec::Q6K,
    9
);

/// Determinism: 20 back-to-back runs of the same `(codec, s)` case must
/// produce byte-identical output -- the fold introduces no new
/// nondeterminism (no atomics, no unordered threadgroup combine) beyond
/// what the plain row-blocked path already has.
#[test]
fn q4k_eight_activation_rows_is_byte_identical_across_twenty_runs() {
    const IN_DIM: usize = 512;
    const OUT_ROWS: usize = 24;
    const TOKENS: usize = 8;

    let rows: Vec<Vec<f32>> = (0..OUT_ROWS)
        .map(|row| random_vec(61 + row as u64, IN_DIM))
        .collect();
    let packed = pack_rows(Codec::Q4K, &rows, IN_DIM);
    let activation = random_vec(97, TOKENS * IN_DIM);
    let (program, sum) = matmul_program(TOKENS as u32, IN_DIM as u32, OUT_ROWS as u32);
    let blocks = [
        QuantizedBlock::Q4K(&packed),
        QuantizedBlock::Float32(&activation),
    ];
    let plan = omega::plan(&program, &[], &blocks, &[sum], NumericPolicy::default()).expect("metal plans the matmul");

    let first =
        omega::execute_plan(&plan, &blocks).expect("metal runs the matmul on a real device");
    let first_bytes: Vec<u8> = first.root().iter().flat_map(|value| value.to_le_bytes()).collect();
    for run in 1..20 {
        let repeat =
            omega::execute_plan(&plan, &blocks).expect("metal runs the matmul on a real device");
        let repeat_bytes: Vec<u8> =
            repeat.root().iter().flat_map(|value| value.to_le_bytes()).collect();
        assert_eq!(
            repeat_bytes, first_bytes,
            "run {run} disagrees byte-for-byte with run 0 on the same plan"
        );
    }
}
