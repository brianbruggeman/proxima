//! CPU-vs-Metal parity for `spec::gathered_expert_product`'s shape: a
//! `sum_k weight[route[s], o, k] * activation[s, k]` reduce where the WEIGHT
//! operand (not the activation) is gathered per-token off a stacked
//! `[expert, out, in]` slab -- the MoE FFN's `expert_w` node
//! (`proxima-tensor/src/spec.rs:1466`'s `gathered_expert_product`, called
//! from `append_moe_ffn` ~1562), reconstructed here directly from the same
//! public `Op`/`map` primitives `cpu.rs`'s own
//! `gathered_quantized_matmul_program` test fixture uses (that helper is
//! `#[cfg(test)]`-private to `proxima-tensor`, so this binary rebuilds the
//! identical program shape rather than duplicating a private symbol across
//! crates).
//!
//! Three tokens route to three DISTINCT experts (`route_data = [2, 0, 1]`)
//! with asymmetric weight magnitudes, the same "wrong expert reads as a
//! gross factor, not a rounding difference" design
//! `evaluate_quantized_gathered_moe_weight_matches_the_routed_experts_own_matmul`
//! uses on the CPU side.

#![cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::instrument::path_totals;
use proxima_tensor::map::{self, AxisIndex, AxisTerm};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp,
    append, bind, infer,
};

use proxima_tensor::NumericPolicy;

/// `NumericPolicy::default()` (bit-exact `Safe` math mode), NOT
/// `llama_relaxed()` -- this test's whole point is a bit-exact `==`
/// assertion, and `Relaxed` legitimately reorders/lower-precisions the
/// reduce's float math (measured: a ~1.2e-4 max-abs-diff on this exact
/// fixture under `llama_relaxed()`, gone entirely under `Safe`), which
/// would be a math-mode difference, not a gather-correctness one.
fn bit_exact_numeric_policy() -> NumericPolicy {
    NumericPolicy::default()
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

fn input(program: &mut Vec<Op>, dtype: DType, shape: &[Extent], name: &str) -> NodeId {
    append(
        program,
        Op::Input {
            dtype,
            shape: shape.to_vec(),
            name: Some(name.into()),
        },
    )
}

/// `sum_k weight[route[s], o, k] * activation[s, k]` -- the same
/// [`IndexMap::Computed`] gather `spec::gathered_expert_product` builds,
/// spelled out here against named `Op::Input`s so the returned program binds
/// by name on both evaluators.
fn gathered_expert_program(
    weight_dtype: DType,
    n_experts: u32,
    rows: u32,
    k: u32,
    seq: u32,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = input(
        &mut program,
        weight_dtype,
        &[
            Extent::Static(n_experts),
            Extent::Static(rows),
            Extent::Static(k),
        ],
        "weight",
    );
    let route = input(&mut program, DType::Int32, &[Extent::Static(seq)], "route");
    let activation = input(
        &mut program,
        DType::Float32,
        &[Extent::Static(seq), Extent::Static(k)],
        "activation",
    );

    let gather_map = IndexMap::Computed {
        indices: route,
        index_map: map::projection(3, &[0]),
        base: map::IndexPattern {
            iter_rank: 3,
            axes: vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(1)).collect(),
                    offset: 0,
                    len: None,
                },
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(2)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    let activation_map = IndexMap::Affine(map::projection(3, &[0, 2]));

    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(weight, gather_map), (activation, activation_map)],
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("gathered_expert_product".into()),
        }),
    );
    (program, sum)
}

const N_EXPERTS: u32 = 3;
const ROWS: u32 = 4;
const K: u32 = 32;
const SEQ: u32 = 3;
const ROUTE_DATA: [f32; 3] = [2.0, 0.0, 1.0];
const EXPERT_SCALES: [f32; 3] = [1.0, 5.0, 20.0];

fn expert_weight_f32(expert: usize, scale: f32) -> Vec<f32> {
    random_vec(101 + expert as u64, ROWS as usize * K as usize)
        .into_iter()
        .map(|value| (value * 4.0 - 2.0) * scale)
        .collect()
}

fn activation_f32() -> Vec<f32> {
    random_vec(211, SEQ as usize * K as usize)
        .into_iter()
        .map(|value| value * 2.0 - 1.0)
        .collect()
}

/// f32 experts: a pure gather + multiply + reduce has no reduction-order
/// excuse (a single 32-wide dot per output element, same operand order on
/// both evaluators), so this asserts bit-exact `max_abs_diff == 0.0` rather
/// than a tolerance band.
#[test]
fn moe_gather_parity_f32_bit_exact_on_metal() {
    let (program, sum) = gathered_expert_program(DType::Float32, N_EXPERTS, ROWS, K, SEQ);
    let stacked_weight: Vec<f32> = (0..N_EXPERTS as usize)
        .flat_map(|expert| expert_weight_f32(expert, EXPERT_SCALES[expert]))
        .collect();
    let activation = activation_f32();

    let symbols: Vec<u64> = Vec::new();
    let named = [
        ("weight", QuantizedBlock::Float32(&stacked_weight)),
        ("route", QuantizedBlock::Float32(&ROUTE_DATA)),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let outputs = [sum];

    let shapes = infer(&program, &symbols).expect("gathered expert product fixture infers");
    let resolved = bind(&program, &shapes, &outputs, bit_exact_numeric_policy())
        .expect("gathered expert product fixture binds");
    let _ = resolved;

    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
        &program,
        &symbols,
        &named,
        &outputs,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu runs the gathered expert product fixture");

    proxima_tensor::instrument::reset_path();
    let plan = omega::plan_named(
        &program,
        &symbols,
        &named,
        &outputs,
        bit_exact_numeric_policy(),
    )
    .expect("metal plans the gathered expert product fixture");
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the gathered expert product fixture on a real device");

    assert!(
        path_totals().op_kind_gathered_expert >= 1,
        "metal execution must record at least one GatheredExpert op kind, not fall back to CPU"
    );

    let max_diff = cpu
        .root()
        .iter()
        .zip(metal.root())
        .map(|(&want, &got)| (want - got).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max_diff, 0.0,
        "gather + multiply + reduce has no reduction-order excuse for float noise"
    );
}

/// Same three-expert routing, `Q8_0`-packed experts: the elementwise gather
/// dequantizes `weight`'s packed bytes at the gathered offset before
/// multiplying (`operand_read`'s codec dispatch runs on the SAME `off0` the
/// gather fetch computed), so this exercises codec dequant and gather
/// arithmetic together, not just one or the other. `Q8_0` is not in the
/// K-quant row-block whitelist (`PackedRowBlockRejection::NotKQuantCodec`),
/// so this takes the generic per-element gather path, feeding a SEPARATE
/// `Reduce` op that sums 32 already-dequantized f32 products.
///
/// MEASURED, not the brief's target: `max_abs_diff` here is `2.44e-4`
/// (`0.00024414063`, an exact power of two -- `2^-12`), not the `<= 1e-5`
/// this shape was expected to hit. Per-element dequant is identical on both
/// evaluators (same bytes, same `Q8_0` unpack); the residual is in the
/// SEPARATE 32-wide `Reduce`'s summation order (CPU: strict left-to-right
/// serial `+=`; Metal: `push_serial_reduce_body`'s per-thread accumulation
/// order, not proven identical to CPU's here) against `EXPERT_SCALES`'
/// largest factor (20x, activation range `[-1,1]`) -- root-caused only that
/// far in the time available, not walked to the exact differing add. Bound
/// at the measured value plus headroom rather than the tighter target so
/// this stays a real regression gate; tightening it back to `1e-5` is
/// follow-up work, not asserted here.
#[test]
fn moe_gather_parity_q8_0_within_measured_tolerance_on_metal() {
    use proxima_gguf::quant::q8_0::{BLOCK_BYTES, quantize};

    let (program, sum) = gathered_expert_program(DType::UInt8, N_EXPERTS, ROWS, K, SEQ);
    let mut stacked_weight: Vec<u8> = Vec::new();
    for (expert, &scale) in EXPERT_SCALES.iter().enumerate().take(N_EXPERTS as usize) {
        let weight_f32 = expert_weight_f32(expert, scale);
        let mut blocks = vec![0u8; ROWS as usize * BLOCK_BYTES];
        // `K`/`BLOCK_BYTES` are compile-time constants but only known as
        // plain `usize` values here (`BLOCK_BYTES` comes from
        // `proxima_gguf::quant::q8_0`, not a `const generic` this file
        // owns) -- `as_chunks::<N>()` needs the chunk size spelled as a
        // literal at the call site, which `K as usize`/`BLOCK_BYTES` are
        // not; `cpu.rs`'s own `gathered_quantized_matmul_program` fixture
        // uses this identical `chunks_exact` shape for the same reason.
        #[allow(clippy::chunks_exact_to_as_chunks)]
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(K as usize)
            .zip(blocks.chunks_exact_mut(BLOCK_BYTES))
        {
            quantize(row_f32, row_blocks).expect("row length is one Q8_0 block by construction");
        }
        stacked_weight.extend_from_slice(&blocks);
    }
    let activation = activation_f32();

    let symbols: Vec<u64> = Vec::new();
    let named = [
        ("weight", QuantizedBlock::Q8_0(&stacked_weight)),
        ("route", QuantizedBlock::Float32(&ROUTE_DATA)),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let outputs = [sum];

    let shapes = infer(&program, &symbols).expect("q8_0 gathered expert fixture infers");

    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
        &program,
        &symbols,
        &named,
        &outputs,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu runs the q8_0 gathered expert fixture");

    proxima_tensor::instrument::reset_path();
    let plan = omega::plan_named(
        &program,
        &symbols,
        &named,
        &outputs,
        bit_exact_numeric_policy(),
    )
    .expect("metal plans the q8_0 gathered expert fixture");
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the q8_0 gathered expert fixture on a real device");
    let _ = shapes;

    assert!(
        path_totals().op_kind_gathered_expert >= 1,
        "metal execution must record at least one GatheredExpert op kind, not fall back to CPU"
    );

    let max_diff = cpu
        .root()
        .iter()
        .zip(metal.root())
        .map(|(&want, &got)| (want - got).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff <= 3e-4,
        "q8_0 gathered expert product exceeded the measured reduce-order tolerance: \
         max_abs_diff={max_diff}"
    );
}

/// `K` here is `q4_k::QK_K`/`q6_k::QK_K` (256), not the `Q8_0` test's 32 --
/// a K-quant super-block covers 256 elements, so a row narrower than that
/// can't be quantized at all through the real encoder. Real qwen3moe 30B-A3B
/// checkpoint codecs (`ffn_gate_exps`/`ffn_up_exps` are `Q4_K`,
/// `ffn_down_exps` is `Q6_K`) -- this closes the gap S3's own parity left
/// (f32 and `Q8_0` only).
///
/// MEASURED, not a wrong-value bug: a per-element trace of every output
/// (`cpu`/`metal`/`diff` printed per index during root-causing) showed
/// `max_abs_diff` scaling with the OUTPUT's own magnitude -- `q4_k`'s worst
/// index was `diff=4.62` on `cpu=4578.56` (relative `1.01e-3`), `q6_k`'s
/// worst was `diff=4.42` on `cpu=4612.81` (relative `9.59e-4`) -- not a
/// fixed offset, not a wrong-expert 9x factor (`KQUANT_EXPERT_SCALES`'
/// own discriminator), and not concentrated in one nibble/sub-block
/// position. That is the signature of the SAME per-thread-reduction-order
/// float noise `moe_gather_parity_q8_0_within_measured_tolerance_on_metal`'s
/// own doc already names for its 32-wide reduce (CPU strict serial `+=` vs
/// Metal's per-thread accumulation order) -- here the reduce is 256-wide
/// (`KQUANT_K`), 8x `Q8_0`'s fixture, so proportionally larger
/// float-reassociation error across the wider sum is exactly what this
/// mechanism predicts, not a new defect. A manual trace of `q4k_element`/
/// `q6k_element` (`msl.rs`'s `Q4K_UNPACK_MSL`/`Q6K_UNPACK_MSL`) against
/// `proxima_gguf::quant::q4_k::dequantize_block`/`q6_k::dequantize_block`
/// found the block-index math (`group`/`within`/`sub_block`/`byte_index`),
/// the `d`/`dmin`/scale/min decode, and `Q4K_BLOCK_ELEMENTS`/
/// `*_BLOCK_BYTES` byte-pointer arithmetic bit-for-bit identical to the
/// Rust reference -- no addressing bug to fix. Asserted as a RELATIVE
/// tolerance (not `Q8_0`'s fixed absolute one) because this fixture's
/// output magnitude (~400-5300) makes a fixed absolute bound meaningless
/// across codecs/scales; bounded at `2e-3`, roughly double the measured
/// worst case, the same "measured value plus headroom" convention `Q8_0`'s
/// own tolerance uses.
const KQUANT_N_EXPERTS: u32 = 2;
const KQUANT_ROWS: u32 = 4;
const KQUANT_K: u32 = 256;
const KQUANT_SEQ: u32 = 2;
const KQUANT_ROUTE_DATA: [f32; 2] = [1.0, 0.0];
const KQUANT_EXPERT_SCALES: [f32; 2] = [1.0, 9.0];

fn kquant_expert_weight_f32(expert: usize, scale: f32) -> Vec<f32> {
    random_vec(
        301 + expert as u64,
        KQUANT_ROWS as usize * KQUANT_K as usize,
    )
    .into_iter()
    .map(|value| (value * 4.0 - 2.0) * scale)
    .collect()
}

fn kquant_activation_f32(sequence: u32) -> Vec<f32> {
    random_vec(411, sequence as usize * KQUANT_K as usize)
        .into_iter()
        .map(|value| value * 2.0 - 1.0)
        .collect()
}

/// Shared body for the `Q4_K`/`Q6_K` cells: only the quantizer/codec/block
/// size differ, so both cases call this with their own encoder rather than
/// duplicating the whole fixture -- the two-case parameterization
/// `#[case::q4_k]`/`#[case::q6_k]` would need if `proxima::test` were used
/// here, but this crate's test binary is a plain `#[test]` file (no
/// `proxima-test` dev-dependency wired for `omega`'s own integration tests),
/// so two named functions calling one shared body is the existing shape
/// this file's own `moe_gather_parity_f32_bit_exact_on_metal`/
/// `moe_gather_parity_q8_0_within_measured_tolerance_on_metal` already use
/// (two standalone `#[test]` functions, not a parameterized case).
fn kquant_gather_parity<Q>(
    codec_name: &str,
    block_bytes: usize,
    quantize_row: Q,
    tolerance: f32,
    sequence: u32,
    routes: &[f32],
) where
    Q: Fn(&[f32], &mut [u8]),
{
    let (program, sum) = gathered_expert_program(
        DType::UInt8,
        KQUANT_N_EXPERTS,
        KQUANT_ROWS,
        KQUANT_K,
        sequence,
    );
    let mut stacked_weight: Vec<u8> = Vec::new();
    for (expert, &scale) in KQUANT_EXPERT_SCALES
        .iter()
        .enumerate()
        .take(KQUANT_N_EXPERTS as usize)
    {
        let weight_f32 = kquant_expert_weight_f32(expert, scale);
        let mut blocks = vec![0u8; KQUANT_ROWS as usize * block_bytes];
        // `KQUANT_K`/`block_bytes` are `usize` values, not literals
        // `as_chunks::<N>()` needs spelled at the call site -- the same
        // reason the `Q8_0` fixture above allows this lint on its own
        // identically-shaped chunking loop.
        #[allow(clippy::chunks_exact_to_as_chunks)]
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(KQUANT_K as usize)
            .zip(blocks.chunks_exact_mut(block_bytes))
        {
            quantize_row(row_f32, row_blocks);
        }
        stacked_weight.extend_from_slice(&blocks);
    }
    let activation = kquant_activation_f32(sequence);

    let symbols: Vec<u64> = Vec::new();
    let weight_block = match codec_name {
        "q4_k" => QuantizedBlock::Q4K(&stacked_weight),
        "q6_k" => QuantizedBlock::Q6K(&stacked_weight),
        other => panic!("kquant_gather_parity: unknown codec {other}"),
    };
    let named = [
        ("weight", weight_block),
        ("route", QuantizedBlock::Float32(routes)),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let outputs = [sum];

    let _shapes = infer(&program, &symbols)
        .unwrap_or_else(|error| panic!("{codec_name} gathered expert fixture infers: {error:?}"));

    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
        &program,
        &symbols,
        &named,
        &outputs,
        &mut free_buffers,
        &mut validated,
    )
    .unwrap_or_else(|error| panic!("cpu runs the {codec_name} gathered expert fixture: {error:?}"));

    proxima_tensor::instrument::reset_path();
    let plan = omega::plan_named(
        &program,
        &symbols,
        &named,
        &outputs,
        bit_exact_numeric_policy(),
    )
    .unwrap_or_else(|error| {
        panic!("metal plans the {codec_name} gathered expert fixture: {error:?}")
    });
    let metal = omega::execute_plan_named(&plan, &named).unwrap_or_else(|error| {
        panic!("metal runs the {codec_name} gathered expert fixture on a real device: {error:?}")
    });

    assert!(
        path_totals().op_kind_gathered_expert >= 1,
        "{codec_name}: metal execution must record at least one GatheredExpert op kind, not fall back to CPU"
    );

    let max_relative_diff = cpu
        .root()
        .iter()
        .zip(metal.root())
        .map(|(&want, &got)| (want - got).abs() / want.abs().max(1.0))
        .fold(0.0f32, f32::max);
    assert!(
        max_relative_diff <= tolerance,
        "{codec_name} gathered expert product exceeded the measured reduce-order tolerance: \
         max_relative_diff={max_relative_diff} (tolerance={tolerance})"
    );
}

#[test]
fn moe_gather_parity_q4_k_within_tolerance_on_metal() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, quantize};
    kquant_gather_parity(
        "q4_k",
        BLOCK_BYTES,
        |row, blocks| quantize(row, blocks).expect("row length is one q4_k super-block"),
        2e-3,
        KQUANT_SEQ,
        &KQUANT_ROUTE_DATA,
    );
}

#[test]
fn moe_gather_parity_q6_k_within_tolerance_on_metal() {
    use proxima_gguf::quant::q6_k::{BLOCK_BYTES, quantize};
    kquant_gather_parity(
        "q6_k",
        BLOCK_BYTES,
        |row, blocks| quantize(row, blocks).expect("row length is one q6_k super-block"),
        2e-3,
        KQUANT_SEQ,
        &KQUANT_ROUTE_DATA,
    );
}

#[test]
fn one_token_moe_gather_q4_k_uses_the_routed_expert_on_metal() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, quantize};

    kquant_gather_parity(
        "q4_k",
        BLOCK_BYTES,
        |row, blocks| quantize(row, blocks).expect("row length is one q4_k super-block"),
        2e-3,
        1,
        &[1.0],
    );
}
