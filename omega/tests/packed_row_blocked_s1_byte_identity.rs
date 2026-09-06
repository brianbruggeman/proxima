//! `push_packed_row_blocked_body`'s `token_total <= 1` arm must emit
//! byte-identical MSL to the pre-fold body -- the invariant the
//! `packed_row_multi_activation` -> `push_packed_row_blocked_body` fold was
//! required to hold. `omega/tests/fixtures/packed_row_blocked_s1_q4k.msl` was
//! captured from `omega::emit` BEFORE that fold, on the exact `(tokens=1,
//! in_dim=512, out_dim=24)` Q4_K matvec `packed_row_multi_activation_parity.rs`'s
//! own `run_case` builds; this test re-emits the identical op today and
//! diffs the source text.

// `metal-q5k-pair-dot` no longer exists as a cargo feature (`omega/Cargo.
// toml`'s own doc): the paired-nibble body is now selected by
// `PackedCodec::supports_pair_dot`, a structural fact of the codec, and only
// ever applies to `Q5_K`/`Q6_K`, never `Q4_K` -- so it was never a legitimate
// member of this exclusion list, and naming it here after its removal would
// make the whole cfg gate unsatisfiable under any real build (unknown
// `cfg(feature = ...)` just evaluates false, it does not error).
//
// `metal-q4k-ggml-port` and `metal-packed-row-nsg2` are no longer excluded:
// both joined `metal`'s own default feature list (`omega/Cargo.toml`), so
// the fixture below IS their emit -- excluding them made this file compile
// to zero tests under every real `--features metal` build (dead since ROW
// 311). The three still excluded are non-default and each provably changes
// this exact op's emitted text: `metal-q4k-split-k` and
// `metal-q4k-single-fetch` re-route `push_packed_row_blocked_body`'s own arm
// selection (`msl.rs`'s `push_q4k_single_fetch_body`/split-K combine arms),
// and `metal-q4k-mask-fma` swaps `push_q4k_header_decode`/
// `push_q4k_product_reduce_body`'s scale-deferred arm for a branch-free
// mask/FMA rewrite. `metal-tiled-gemm` is left OFF this list on purpose --
// read, not assumed: `classify_tiled_gemm` (`msl.rs`) rejects any op below
// `TILED_GEMM_MIN_TOKENS` regardless of whether the feature is compiled in,
// and `TOKENS = 1` here is always below it, so enabling the feature cannot
// change this op's emitted body -- it only unlocks a code path this fixture
// never reaches.
#![cfg(all(
    feature = "metal",
    target_os = "macos",
    not(any(
        feature = "metal-q4k-split-k",
        feature = "metal-q4k-single-fetch",
        feature = "metal-q4k-mask-fma",
    ))
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append, projection,
};

const QK_K: usize = 256;

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = proxima_tensor::test_support::Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

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

/// A single activation row (`token_total == 1`) must take
/// `push_packed_row_blocked_body`'s unchanged, pre-fold arm -- proven by
/// byte-identity against the fixture, not just "the test suite still
/// passes" (a well-tested wrong emission is harder to see than a failing
/// one). The whole file is gated off every OTHER feature that legitimately
/// changes this same body's emitted text regardless of `token_total` (see
/// the module's own `#![cfg(..)]`) -- the fixture is a snapshot of the
/// DEFAULT-feature body, not every combination, the same scoping
/// `packed_row_multi_activation_parity.rs`'s own `token_group` marker
/// assertion applies against `metal-tiled-gemm`.
///
/// A second assertion checks the fixture is not merely SOME body, but the
/// DEFAULT one: `metal-q4k-ggml-port` and `metal-packed-row-nsg2` sit inside
/// `metal`'s own default feature list, so a build that silently drops either
/// out of that default list would still pass byte-identity against a STALE
/// fixture forever (the fixture would simply have been captured from the old
/// default too) -- checking for `push_q4k_ggml_port_body`'s own `sc16_0`/
/// `acc1_0` identifiers in the source text, plus the dispatched nsg=2 width
/// on `kernel.grid` (`tiled_gemm_threadgroup_width`'s own
/// `SIMD_WIDTH * packed_row_nsg_factor()`, both from `omega::sized` so this
/// assertion moves if the runtime config ever does), makes a silent default
/// change fail loudly here instead of only showing up as a decode-step
/// regression weeks later.
#[test]
fn single_activation_row_q4k_matvec_emits_byte_identical_msl_to_the_pre_fold_body() {
    const IN_DIM: usize = 512;
    const OUT_ROWS: usize = 24;
    const TOKENS: usize = 1;

    let rows: Vec<Vec<f32>> = (0..OUT_ROWS)
        .map(|row| random_vec(61 + row as u64, IN_DIM))
        .collect();
    let blocks_per_row = IN_DIM / QK_K;
    let mut packed = vec![0u8; OUT_ROWS * blocks_per_row * proxima_gguf::quant::q4_k::BLOCK_BYTES];
    for (row, row_packed) in rows
        .iter()
        .zip(packed.chunks_exact_mut(blocks_per_row * proxima_gguf::quant::q4_k::BLOCK_BYTES))
    {
        proxima_gguf::quant::q4_k::quantize(row, row_packed).unwrap();
    }

    let (program, sum) = matmul_program(TOKENS as u32, IN_DIM as u32, OUT_ROWS as u32);
    let shapes = proxima_tensor::infer(&program, &[]).expect("the synthetic program infers");
    let packed_operands: omega::PackedOperands =
        [(NodeId(0), omega::PackedCodec::Q4K)].into_iter().collect();
    let mut bound =
        proxima_tensor::bind(&program, &shapes, &[sum]).expect("the synthetic program binds");
    proxima_tensor::correct_packed_matmul_layouts(&mut bound, &[NodeId(0)].into_iter().collect());
    let resolved = bound
        .iter()
        .find(|op| op.node == sum)
        .expect("the reduce node is bound");
    let kernel = omega::emit(resolved, &packed_operands).expect("the synthetic program emits");

    let expected = include_str!("fixtures/packed_row_blocked_s1_q4k.msl");
    assert_eq!(
        kernel.source, expected,
        "s=1 Q4_K matvec MSL drifted from the pre-fold fixture -- the \
         `token_total <= 1` branch must stay byte-identical to the body \
         captured before `push_packed_row_multi_activation_body` folded into \
         `push_packed_row_blocked_body`"
    );

    assert!(
        kernel.source.contains("sc16_0") && kernel.source.contains("acc1_0"),
        "the fixture no longer carries `push_q4k_ggml_port_body`'s own \
         `sc16_0`/`acc1_0` identifiers -- `metal-q4k-ggml-port` sits in \
         `metal`'s own default feature list, so its body must be what this \
         fixture captures; a build that silently dropped it from the \
         default would still byte-match a stale fixture without this check"
    );
    let expected_nsg2_width = omega::sized::SIMD_WIDTH * omega::sized::PACKED_ROW_NSG as u64;
    assert_eq!(
        kernel.grid.threadgroup_width,
        Some(expected_nsg2_width),
        "the dispatched threadgroup width drifted from `SIMD_WIDTH * \
         PACKED_ROW_NSG` -- `metal-packed-row-nsg2`/`metal-q4k-ggml-port`'s \
         shared nsg=2 geometry (`tiled_gemm_threadgroup_width`'s own \
         `packed_row_nsg_factor`) must still be what a default build \
         dispatches"
    );
}
