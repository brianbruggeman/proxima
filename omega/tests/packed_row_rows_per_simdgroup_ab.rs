//! Intervention 5 (`RUN.md`, section "Intervention 5"): the packed-row-
//! blocked reduce renderer's rows-per-simdgroup count is now a runtime
//! parameter (`PROXIMA_PACKED_ROWS=8`, default 4 = today's emit --
//! `omega/src/msl/kernel_types_identity.rs`'s `codec_rows_per_simdgroup`).
//! This gate proves widening the row group changes no arithmetic: for every
//! shape below, `rows=8` must produce BIT-EXACT (`to_bits()`) output
//! against `rows=4` on the same real-data-shaped Q4_0 weights/activations --
//! not merely close, since widening only adds independent per-lane row
//! chains, it must never reorder an existing row's accumulation
//! (`push_packed_row_blocked_body`'s own doc on the change).

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

/// `[out_dim, in_dim] x [tokens=1, in_dim] -> [1, out_dim]` -- a plain
/// matvec, the same shape family `packed_row_multi_activation_parity.rs`
/// builds, single activation row so the plain row-blocked arm (not the
/// multi-activation fold) is what dispatches.
fn matvec_program(in_dim: u32, out_dim: u32) -> (Vec<Op>, NodeId) {
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
            shape: vec![Extent::Static(1), Extent::Static(in_dim)],
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
    let mut packed = vec![0u8; rows.len() * blocks_per_row * proxima_gguf::quant::q4_0::BLOCK_BYTES];
    for (row, row_packed) in rows
        .iter()
        .zip(packed.chunks_exact_mut(blocks_per_row * proxima_gguf::quant::q4_0::BLOCK_BYTES))
    {
        proxima_gguf::quant::q4_0::quantize(row, row_packed).unwrap();
    }
    packed
}

/// Runs the matvec once under the given `PROXIMA_PACKED_ROWS` env value
/// (`None` = unset, today's default) and returns the raw output bit
/// patterns -- `to_bits()`, not the floats, so a caller compares for exact
/// equality rather than a tolerance band.
fn run_bits(in_dim: usize, out_dim: usize, seed: u64, packed_rows_env: Option<&str>) -> Vec<u32> {
    let rows: Vec<Vec<f32>> = (0..out_dim)
        .map(|row| random_vec(seed + row as u64, in_dim))
        .collect();
    let packed = pack_rows(&rows, in_dim);
    let activation = random_vec(seed + 9973, in_dim);
    let (program, sum) = matvec_program(in_dim as u32, out_dim as u32);
    let blocks = [
        QuantizedBlock::Packed { codec: omega::Codec::Q4_0, bytes: &packed },
        QuantizedBlock::Float32(&activation),
    ];

    let output = temp_env::with_var("PROXIMA_PACKED_ROWS", packed_rows_env, || {
        let plan = omega::plan(&program, &[], &blocks, &[sum], NumericPolicy::default())
            .expect("metal plans the matvec");
        omega::execute_plan(&plan, &blocks).expect("metal runs the matvec on a real device")
    });
    output.root().iter().map(|value| value.to_bits()).collect()
}

/// One `(K, rows)` case: `rows=8` must reproduce `rows=4`'s output
/// bit-for-bit. `rows` here is the OUTPUT row count (`out_dim`), matching
/// the brief's naming; each lane's per-row accumulation order is unchanged
/// by widening (same `q4_0_pair_dot` sequence, same `simd_sum`), so any
/// bit divergence means the widening reordered arithmetic, not merely lost
/// precision.
fn assert_rows4_and_rows8_bit_exact(k: usize, rows: usize, seed: u64) {
    let baseline = run_bits(k, rows, seed, None);
    let widened = run_bits(k, rows, seed, Some("8"));
    assert_eq!(
        baseline.len(),
        rows,
        "degenerate gate: K={k} rows={rows} baseline produced the wrong element count"
    );
    assert_same_accumulation(k, rows, &baseline, &widened);
}

#[cfg(not(feature = "metal-q4k-split-k"))]
fn assert_same_accumulation(k: usize, rows: usize, baseline: &[u32], widened: &[u32]) {
    assert_eq!(
        baseline, widened,
        "K={k} rows={rows}: PROXIMA_PACKED_ROWS=8 produced different bits than the rows=4 \
         default -- widening rows-per-simdgroup must not reorder any row's accumulation"
    );
}

/// Under split-k the K-axis partition is `TARGET_SIMDGROUPS / (rows /
/// rows_per_group)` simdgroups wide, so changing rows-per-group changes the
/// split factor (K=6144, rows=1536: 2048/384 = 5 at rows=4, 2048/192 clamped
/// to 8 at rows=8) and with it the order a row's partial sums combine in.
/// The contract there is agreement to reassociation error, not identical bits.
#[cfg(feature = "metal-q4k-split-k")]
fn assert_same_accumulation(k: usize, rows: usize, baseline: &[u32], widened: &[u32]) {
    const RELATIVE_TOLERANCE: f32 = 1e-5;
    for (index, (&want_bits, &got_bits)) in baseline.iter().zip(widened).enumerate() {
        let (want, got) = (f32::from_bits(want_bits), f32::from_bits(got_bits));
        let allowed = RELATIVE_TOLERANCE * want.abs().max(got.abs()).max(1.0);
        assert!(
            (want - got).abs() <= allowed,
            "K={k} rows={rows} element {index}: rows=4 gave {want}, rows=8 gave {got}, beyond \
             the {RELATIVE_TOLERANCE} reassociation tolerance"
        );
    }
}

#[test]
fn k1536_rows2048_bit_exact_across_row_group_widths() {
    assert_rows4_and_rows8_bit_exact(1536, 2048, 11);
}

#[test]
fn k2048_rows1536_bit_exact_across_row_group_widths() {
    assert_rows4_and_rows8_bit_exact(2048, 1536, 23);
}

#[test]
fn k1536_rows6144_bit_exact_across_row_group_widths() {
    assert_rows4_and_rows8_bit_exact(1536, 6144, 37);
}

#[test]
fn k1536_rows256_bit_exact_across_row_group_widths() {
    assert_rows4_and_rows8_bit_exact(1536, 256, 53);
}

#[test]
fn k6144_rows1536_bit_exact_across_row_group_widths() {
    assert_rows4_and_rows8_bit_exact(6144, 1536, 71);
}

/// Zero-diff proof: with `PROXIMA_PACKED_ROWS` unset, the default emit for
/// node 94's own shape (`K=1536`, `Q4_0`, plain-product reduce) renders to
/// `<rep>/intervention5/emitted_node94_shape.metal` -- diffed by the run
/// record against the real captured production kernel
/// (`<rep>/capture/pipeline_9e2c4d844cac82c2.metal`), not asserted by a
/// hand-copied string literal, so the proof is an actual `diff`, not a
/// second transcription that could itself drift from either source.
#[test]
fn default_env_emits_node_94_shape_for_diffing_against_the_capture() {
    const IN_DIM: usize = 1536;
    const OUT_ROWS: usize = 2048;
    let rows: Vec<Vec<f32>> = (0..OUT_ROWS)
        .map(|row| random_vec(61 + row as u64, IN_DIM))
        .collect();
    let packed = pack_rows(&rows, IN_DIM);
    let (program, sum) = matvec_program(IN_DIM as u32, OUT_ROWS as u32);
    let shapes = proxima_tensor::infer(&program, &[]).expect("the synthetic program infers");
    let packed_operands: omega::PackedOperands =
        [(NodeId(0), omega::Codec::Q4_0)].into_iter().collect();
    let mut bound = proxima_tensor::bind(&program, &shapes, &[sum], NumericPolicy::default())
        .expect("the synthetic program binds");
    proxima_tensor::correct_packed_matmul_layouts(&mut bound, &[NodeId(0)].into_iter().collect());
    let resolved = bound
        .iter()
        .find(|op| op.node == sum)
        .expect("the reduce node is bound");

    let source = temp_env::with_var("PROXIMA_PACKED_ROWS", None::<&str>, || {
        omega::emit(resolved, &packed_operands, NumericPolicy::default())
            .expect("the synthetic program emits")
            .source
    });
    let _ = &packed;

    if cfg!(feature = "metal-q4_0-native") {
        assert!(
            source.contains("sumy * -8.0f") && !source.contains("q4_0_pair_dot(blk"),
            "node 94's shape must take ggml's inline Q4_0 dot under metal-q4_0-native:\n{source}"
        );
    } else {
        assert!(
            source.contains("q4_0_pair_dot(blk"),
            "node 94's shape must take the batched Q4_0 pair-dot arm:\n{source}"
        );
    }
    let scratch = tempfile::tempdir().expect("scratch dir creates");
    let out_dir = std::env::var_os("PROXIMA_EMITTED_MSL_OUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| scratch.path().to_path_buf());
    std::fs::create_dir_all(&out_dir).expect("output dir creates");
    let out_path = out_dir.join("emitted_node94_shape.metal");
    std::fs::write(&out_path, &source).expect("emitted source writes to the output dir");
}

/// Tail case: `rows=1540` is not a multiple of 8 (or of 4) -- proves the
/// `ceil(rows/8)` grid still covers every row and the tail threadgroups'
/// extra idle lanes (rows 1537..1540 alone in their group of 8) do not
/// corrupt output for the rows that DO share a group.
#[test]
fn k1536_rows1540_tail_not_divisible_by_eight_bit_exact() {
    assert_rows4_and_rows8_bit_exact(1536, 1540, 89);
}
