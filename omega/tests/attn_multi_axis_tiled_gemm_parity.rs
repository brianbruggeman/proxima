//! ROW 114's own correctness gate: `q4k_matmul_layout.rs` proves ONE
//! weight-owned output axis at tile scale; the attention Q/K/V/attn_output
//! projections keep TWO (`heads`, `head_dim` -- `wq`'s own declared shape in
//! `proxima_tensor::spec::mistral_cached_forward_program`), folded into one
//! flattened feature dimension by [`classify_tiled_gemm`]'s generalized
//! axis-group check. This is the shape that check newly admits, checked at
//! tile scale (many tokens, `heads*head_dim` NOT a whole multiple of
//! `TILED_GEMM_BLOCK_M`) against an INDEPENDENT reference -- not just CPU
//! against Metal, so the two backends agreeing on the same wrong axis
//! assignment cannot pass this gate. Mirrors the module doc's own warning:
//! a first attempt at a related axis-order bug measured `relative=0.497`,
//! dimensionally valid MSL, semantically wrong -- only an independent
//! reference caught it.

#![cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
use proxima_tensor::cpu::evaluate_quantized;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, projection};

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// `[tokens, in_dim] x [in_dim, heads, head_dim] -> [tokens, heads,
/// head_dim]`, reduced over `in_dim` -- the exact iteration shape
/// `append_mistral_cached_layer`'s own `q_product`/`q` reduce takes
/// (`"si->shdi"` then `"shdi->shd"`, with the einsum's `s`/`h`/`d`/`i`
/// relabeled `tok`/`head`/`hd`/`in` here), generalized to `tokens > 1` so
/// the tiled path's `TILED_GEMM_MIN_TOKENS` gate actually opens. Axis order
/// (tok=0, in=1, head=2, hd=3) puts the reduce axis where the real spec
/// puts it -- second, between the token axis and the feature axes -- so
/// this also re-covers `q4k_matmul_layout.rs`'s "declared reduction axis
/// first within the weight's own operand" shape, one level up.
fn multi_axis_matmul_program(tokens: u32, in_dim: u32, heads: u32, head_dim: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![Extent::Static(in_dim), Extent::Static(heads), Extent::Static(head_dim)],
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
    // iteration space (tok, in, head, hd): weight reads (in, head, hd),
    // ignoring tok; activation reads (tok, in), ignoring head/hd.
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(4, &[1, 2, 3]))),
                (activation, IndexMap::Affine(projection(4, &[0, 1]))),
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
            in_map: IndexMap::Affine(projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(projection(4, &[0, 2, 3])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

/// Packs `heads * head_dim` independent rows of `in_dim` elements each into
/// GGUF's native `[out_dim, in_dim]` row-major byte layout -- same helper
/// shape as `q4k_matmul_layout.rs`'s `pack_rows`, duplicated here for the
/// same reason that file gives its own duplicates (a standalone test binary,
/// a few lines, not worth a shared dependency for).
fn pack_rows(rows: &[Vec<f32>], in_dim: usize) -> Vec<u8> {
    let blocks_per_row = in_dim / QK_K;
    let mut packed = vec![0u8; rows.len() * blocks_per_row * BLOCK_BYTES];
    for (row, row_packed) in rows.iter().zip(packed.chunks_exact_mut(blocks_per_row * BLOCK_BYTES)) {
        quantize(row, row_packed).expect("in_dim is a whole multiple of QK_K");
    }
    packed
}

/// Dequantizes `packed` row by row and dots each row against every token's
/// activation row -- computed independently of both `proxima_tensor::cpu`'s
/// quantized matmul path and omega's Metal emitter, same independence
/// `q4k_matmul_layout.rs`'s own `expected_output` holds both backends to.
fn expected_output(packed: &[u8], in_dim: usize, out_rows: usize, tokens: usize, activation: &[f32]) -> Vec<f32> {
    let blocks_per_row = in_dim / QK_K;
    let mut dequantized_rows: Vec<Vec<f32>> = Vec::with_capacity(out_rows);
    for row_packed in packed.chunks_exact(blocks_per_row * BLOCK_BYTES) {
        let mut row = vec![0.0f32; in_dim];
        dequantize(row_packed, &mut row).expect("packed row dequantizes");
        dequantized_rows.push(row);
    }
    assert_eq!(dequantized_rows.len(), out_rows, "degenerate fixture: one row per feature element");

    // output iteration order matches `out_map`'s (tok, head, hd) -- tok
    // outermost, out_rows (the flattened head/hd feature index) innermost.
    let mut expected = vec![0.0f32; tokens * out_rows];
    for (token_index, token_activation) in activation.chunks_exact(in_dim).enumerate() {
        for (row_index, row) in dequantized_rows.iter().enumerate() {
            let dot: f32 = row.iter().zip(token_activation.iter()).map(|(weight, value)| weight * value).sum();
            expected[token_index * out_rows + row_index] = dot;
        }
    }
    expected
}

#[test]
fn metal_takes_the_tiled_path_and_agrees_with_the_independent_reference_on_a_two_axis_feature_group() {
    const IN_DIM: usize = 512;
    const HEADS: usize = 3;
    const HEAD_DIM: usize = 8;
    const OUT_ROWS: usize = HEADS * HEAD_DIM;
    // NEITHER `TOKENS` nor `OUT_ROWS` is a whole multiple of its own tile
    // dimension (`TILED_GEMM_BLOCK_N`=32, `TILED_GEMM_BLOCK_M`=64) -- a wrong
    // boundary-tile mask on either axis, or a wrong flattened stride for the
    // two-axis feature group, shows up as a real numeric disagreement here.
    const TOKENS: usize = 20;

    let rows: Vec<Vec<f32>> = (0..OUT_ROWS).map(|row| random_vec(61 + row as u64, IN_DIM)).collect();
    let packed = pack_rows(&rows, IN_DIM);
    let activation = random_vec(97, TOKENS * IN_DIM);
    let expected = expected_output(&packed, IN_DIM, OUT_ROWS, TOKENS, &activation);

    let (program, sum) = multi_axis_matmul_program(TOKENS as u32, IN_DIM as u32, HEADS as u32, HEAD_DIM as u32);
    let blocks = [QuantizedBlock::Q4K(&packed), QuantizedBlock::Float32(&activation)];

    // the correctness claim is hollow if this shape silently stayed on the
    // row-blocked path -- confirm the emitted kernel source actually took
    // the `simdgroup_matrix`-tiled body, the same call-site-marker technique
    // `real_forward_packed_probe.rs` uses for the row-blocked/generic split.
    let shapes = proxima_tensor::infer(&program, &[]).expect("the synthetic program infers");
    let packed_operands: omega::PackedOperands = [(NodeId(0), omega::PackedCodec::Q4K)].into_iter().collect();
    let mut bound = proxima_tensor::bind(&program, &shapes, &[sum]).expect("the synthetic program binds");
    proxima_tensor::correct_packed_matmul_layouts(&mut bound, &[NodeId(0)].into_iter().collect());
    let resolved = bound.iter().find(|op| op.node == sum).expect("the reduce node is bound");
    let kernel = omega::emit(resolved, &packed_operands).expect("the synthetic program emits");
    assert!(
        kernel.source.contains("simdgroup_multiply_accumulate"),
        "degenerate gate: this shape must take the tiled-gemm path, not row-blocked or generic \
         (kernel source did not contain the tiled path's own simdgroup_matrix call)"
    );

    let cpu = evaluate_quantized(&program, &[], &blocks, &[sum]).expect("cpu runs the matmul");
    let plan = omega::plan(&program, &[], &blocks, &[sum]).expect("metal plans the matmul");
    let metal = omega::execute_plan(&plan, &blocks).expect("metal runs the matmul on a real device");

    let cpu_root = cpu.root();
    let metal_root = metal.root();
    let element_count = TOKENS * OUT_ROWS;
    assert_eq!(cpu_root.len(), element_count, "degenerate gate: cpu produced no output");
    assert_eq!(metal_root.len(), element_count, "degenerate gate: metal produced no output");

    let mut max_diff = 0.0f32;
    for (index, ((&cpu_value, &metal_value), &reference)) in
        cpu_root.iter().zip(metal_root.iter()).zip(expected.iter()).enumerate()
    {
        let scale = reference.abs().max(f32::MIN_POSITIVE);
        let cpu_relative = (cpu_value - reference).abs() / scale;
        let metal_relative = (metal_value - reference).abs() / scale;
        max_diff = max_diff.max(metal_relative);
        assert!(
            cpu_relative < 1e-2,
            "element {index}: cpu={cpu_value} disagrees with the independent dequantize+dot reference={reference} \
             (relative={cpu_relative})"
        );
        assert!(
            metal_relative < 5e-3,
            "element {index}: metal={metal_value} disagrees with the independent dequantize+dot reference={reference} \
             (relative={metal_relative}) -- this is the two-axis feature-group fold defect if it fires"
        );
    }
    eprintln!("attn-shaped tiled-gemm metal vs independent reference: max_relative={max_diff}");
}

/// The codegen-level counterpart of ROW 114's own falsifiable criterion:
/// a real per-token DECODE step (`tokens=1`, the shape every
/// `mistral_cached_forward_program` decode call actually takes for
/// `attn_q`/`attn_k`/`attn_v`/`attn_output`) must stay on the row-blocked
/// `q4k_run8` vector path, never the `simdgroup_matrix`-tiled one --
/// `classify_tiled_gemm`'s `TokenExtentBelowMinimum` gate exists precisely
/// so the tiled path's per-tile fixed overhead is never paid on a
/// single-token dispatch. Same shape as the tile-scale test above with
/// `TOKENS` dropped to 1 -- proven from the emitted kernel SOURCE, not
/// inferred from `classify_tiled_gemm`'s own logic.
#[test]
fn metal_decode_shaped_attention_matmul_stays_on_the_row_blocked_vector_path() {
    const IN_DIM: usize = 512;
    const HEADS: usize = 3;
    const HEAD_DIM: usize = 8;
    const TOKENS: usize = 1;

    let (program, sum) = multi_axis_matmul_program(TOKENS as u32, IN_DIM as u32, HEADS as u32, HEAD_DIM as u32);
    let shapes = proxima_tensor::infer(&program, &[]).expect("the synthetic program infers");
    let packed_operands: omega::PackedOperands = [(NodeId(0), omega::PackedCodec::Q4K)].into_iter().collect();
    let mut bound = proxima_tensor::bind(&program, &shapes, &[sum]).expect("the synthetic program binds");
    proxima_tensor::correct_packed_matmul_layouts(&mut bound, &[NodeId(0)].into_iter().collect());
    let resolved = bound.iter().find(|op| op.node == sum).expect("the reduce node is bound");
    let kernel = omega::emit(resolved, &packed_operands).expect("the synthetic program emits");

    assert!(
        !kernel.source.contains("simdgroup_multiply_accumulate"),
        "decode-shaped (tokens=1) attention matmul must NOT take the tiled-gemm path -- \
         the packed weight has already been streamed once per token above the minimum, \
         and paying tile setup for one column is a pure loss"
    );
    assert!(
        kernel.source.contains("q4k_run8"),
        "decode-shaped attention matmul must take the row-blocked vector path \
         (kernel source did not contain the row-blocked path's own q4k_run8 call)"
    );
}
