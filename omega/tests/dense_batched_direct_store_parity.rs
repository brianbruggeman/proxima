//! Byte-identity gate for `PROXIMA_TILED_GEMM_DIRECT_STORE` on
//! [`omega::msl::push_dense_batched_gemm_body`]'s own arm (see
//! `docs/model-interop/discipline.md` ROW C4.11) -- the batched Q.K^T/P.V dense path, distinct from
//! `tiled_gemm_direct_store_parity.rs`'s Q4_0-weighted coverage. Both
//! operands unquantized F32, with a genuine batch axis threaded through
//! `direct_coord[axis] = dense_batch_coord_axis` -- the one code path this
//! lever's Q4_0 coverage cannot exercise at all (Q4_0 tiled-GEMM never
//! carries a batch axis).
//!
//! The output's iteration order is chosen so the FEATURE axis, not the
//! batch axis, is the fastest (last-listed in `out_map`) -- the
//! `direct_store_interior` runtime gate only admits the direct arm when
//! `u.out_strides[feature_axis] == 1` (`push_tiled_gemm_direct_store_arm`'s
//! own doc), so this shape is the one that actually drives the fast arm at
//! runtime rather than always falling back to the restage path.

#![cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit() * 2.0 - 1.0).collect()
}

/// Iteration dims: 0=token, 1=reduce, 2=batch, 3=feature. `weight` reads
/// (feature, batch, reduce); `other` (the token-owned operand) reads
/// (token, batch, reduce); `out_map` lists (token, batch, feature) -- with
/// FEATURE last, so it is the output's fastest axis (this module's own
/// doc), unlike `dense_batched_score_shaped_op` in `omega/src/msl/tests.rs`
/// (batch listed last there).
fn dense_batched_feature_fastest_program(
    token: u32,
    reduce_len: u32,
    batch: u32,
    feature: u32,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(feature),
                Extent::Static(batch),
                Extent::Static(reduce_len),
            ],
            name: None,
        },
    );
    let other = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(token),
                Extent::Static(batch),
                Extent::Static(reduce_len),
            ],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(4, &[3, 2, 1]))),
                (other, IndexMap::Affine(projection(4, &[0, 2, 1]))),
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

fn check_dense_direct_store_byte_identity(
    token: u32,
    reduce_len: u32,
    batch: u32,
    feature: u32,
    label: &str,
) {
    let (program, sum) = dense_batched_feature_fastest_program(token, reduce_len, batch, feature);
    let weight = random_vec(7001 + u64::from(feature), (feature * batch * reduce_len) as usize);
    let other = random_vec(8002 + u64::from(token), (token * batch * reduce_len) as usize);
    let blocks = [QuantizedBlock::Float32(&weight), QuantizedBlock::Float32(&other)];

    let direct_store_off = temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_DIRECT_STORE", None::<&str>),
            ("PROXIMA_TILED_GEMM_DENSE", None::<&str>),
        ],
        || {
            omega::execute(&program, &[], &blocks, &[sum], NumericPolicy::default())
                .expect("metal executes the restage-only dense-batched matmul")
        },
    );
    let direct_store_on = temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_DIRECT_STORE", Some("1")),
            ("PROXIMA_TILED_GEMM_DENSE", None::<&str>),
        ],
        || {
            omega::execute(&program, &[], &blocks, &[sum], NumericPolicy::default())
                .expect("metal executes the direct-store dense-batched matmul")
        },
    );

    let off_root = direct_store_off.root();
    let on_root = direct_store_on.root();
    let element_count = (token * batch * feature) as usize;
    assert_eq!(off_root.len(), element_count, "degenerate: OFF produced no output");
    assert_eq!(on_root.len(), element_count, "degenerate: ON produced no output");

    let mut differing = 0usize;
    let mut max_abs_diff = 0.0f32;
    for (&off_value, &on_value) in off_root.iter().zip(on_root.iter()) {
        if off_value.to_bits() != on_value.to_bits() {
            differing += 1;
        }
        max_abs_diff = max_abs_diff.max((off_value - on_value).abs());
    }
    eprintln!(
        "dense_batched {label} token={token} reduce_len={reduce_len} batch={batch} feature={feature}: \
         direct_store on-vs-off differing_words={differing}/{element_count} max_abs_diff={max_abs_diff}"
    );

    assert_eq!(
        off_root, on_root,
        "dense_batched {label} token={token} feature={feature} batch={batch}: \
         PROXIMA_TILED_GEMM_DIRECT_STORE=1 must produce BIT-IDENTICAL output to the restage-only \
         path -- {differing}/{element_count} words differed, max_abs_diff={max_abs_diff}"
    );
}

/// `feature=128` is a whole multiple of `TILED_GEMM_BLOCK_M` (64) and
/// `token=64` a whole multiple of `TILED_GEMM_BLOCK_N` (32) -- every tile in
/// the dispatch grid is interior, so this is the shape the fast arm is
/// meant to win on, with a real (`batch=4`) batch axis threaded through.
#[test]
fn dense_direct_store_matches_restage_path_at_a_fully_interior_batched_shape() {
    check_dense_direct_store_byte_identity(64, 64, 4, 128, "fully_interior");
}

/// `feature=100` is NOT a multiple of `TILED_GEMM_BLOCK_M` (64) --
/// `100 % 64 == 36` -- forcing the last row-tile of every batch slice
/// through the runtime restage fallback even with the switch on.
#[test]
fn dense_direct_store_matches_restage_path_with_a_partial_edge_tile() {
    check_dense_direct_store_byte_identity(64, 64, 4, 100, "partial_edge_tile");
}

/// Iteration dims: 0=token, 1=reduce, 2=feature, 3=batch. `weight` reads
/// (feature, batch, reduce); `other` (the token-owned operand) reads
/// (token, batch, reduce); `out_map` lists (token, feature, batch) -- with
/// BATCH last, so it is the output's fastest axis, matching this crate's own
/// real gemma4 `score_even`/`score_odd` production layout: measured, real
/// prefill, `out_stride_feature=8` `out_stride_batch=[8, 1]` (see
/// `docs/model-interop/discipline.md` ROW C4.13) -- neither the feature nor the token axis is
/// unit-stride there, only the innermost BATCH axis is, which `simdgroup_store`
/// (row-stride-only, always-contiguous-column hardware intrinsic --
/// `push_tiled_gemm_direct_store_arm`'s own doc) cannot address directly.
/// This shape exists so `PROXIMA_TILED_GEMM_DIRECT_STORE=1` is proven to
/// decline safely (never a wrong fast-path engagement) rather than merely
/// assumed to, on the copy-out loop ROW C4.13 rewrote.
fn dense_batched_batch_innermost_program(
    token: u32,
    reduce_len: u32,
    feature: u32,
    batch: u32,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(feature),
                Extent::Static(batch),
                Extent::Static(reduce_len),
            ],
            name: None,
        },
    );
    let other = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(token),
                Extent::Static(batch),
                Extent::Static(reduce_len),
            ],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(4, &[2, 3, 1]))),
                (other, IndexMap::Affine(projection(4, &[0, 3, 1]))),
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

fn check_dense_batch_innermost_byte_identity(
    token: u32,
    reduce_len: u32,
    feature: u32,
    batch: u32,
    label: &str,
) {
    let (program, sum) = dense_batched_batch_innermost_program(token, reduce_len, feature, batch);
    let weight = random_vec(9001 + u64::from(feature), (feature * batch * reduce_len) as usize);
    let other = random_vec(9502 + u64::from(token), (token * batch * reduce_len) as usize);
    let blocks = [QuantizedBlock::Float32(&weight), QuantizedBlock::Float32(&other)];

    let direct_store_off = temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_DIRECT_STORE", None::<&str>),
            ("PROXIMA_TILED_GEMM_DENSE", None::<&str>),
        ],
        || {
            omega::execute(&program, &[], &blocks, &[sum], NumericPolicy::default())
                .expect("metal executes the restage-only batch-innermost dense-batched matmul")
        },
    );
    let direct_store_on = temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_DIRECT_STORE", Some("1")),
            ("PROXIMA_TILED_GEMM_DENSE", None::<&str>),
        ],
        || {
            omega::execute(&program, &[], &blocks, &[sum], NumericPolicy::default())
                .expect("metal executes the direct-store-requested batch-innermost dense-batched matmul")
        },
    );

    let off_root = direct_store_off.root();
    let on_root = direct_store_on.root();
    let element_count = (token * feature * batch) as usize;
    assert_eq!(off_root.len(), element_count, "degenerate: OFF produced no output");
    assert_eq!(on_root.len(), element_count, "degenerate: ON produced no output");

    let mut differing = 0usize;
    let mut max_abs_diff = 0.0f32;
    for (&off_value, &on_value) in off_root.iter().zip(on_root.iter()) {
        if off_value.to_bits() != on_value.to_bits() {
            differing += 1;
        }
        max_abs_diff = max_abs_diff.max((off_value - on_value).abs());
    }
    eprintln!(
        "dense_batched {label} token={token} reduce_len={reduce_len} feature={feature} batch={batch}: \
         direct_store on-vs-off differing_words={differing}/{element_count} max_abs_diff={max_abs_diff}"
    );

    assert_eq!(
        off_root, on_root,
        "dense_batched {label} token={token} feature={feature} batch={batch}: \
         restage path (with the rewritten copy-out loop) must stay BIT-IDENTICAL regardless of \
         PROXIMA_TILED_GEMM_DIRECT_STORE -- {differing}/{element_count} words differed, \
         max_abs_diff={max_abs_diff}"
    );
}

/// Real gemma4-E2B hippo-prompt (26 real tokens) prefill shape for
/// `score_even`/`score_odd`, MEASURED via `dstore/prompt4_strides_step0.log`
/// (`node=139`): `feature_extent=32 token_extent=27 batch_extent=8
/// reduction_k=128`. `feature=32 < TILED_GEMM_BLOCK_M(64)` and
/// `token=27 < TILED_GEMM_BLOCK_N(32)` -- every tile is a boundary tile at
/// this prompt length, so this also exercises the restage loop's masked
/// (`o_feat < feature_extent && o_tok < token_extent`) branch throughout.
#[test]
fn dense_batch_innermost_matches_restage_path_at_the_real_hippo_prefill_shape() {
    check_dense_batch_innermost_byte_identity(27, 128, 32, 8, "hippo_score_shape");
}

/// Real gemma4-E2B weather-prompt (562 real tokens) prefill shape for
/// `score_even`/`score_odd`, MEASURED via `dstore/prompt5_strides_step0.log`
/// (`node=139`): `feature_extent=576 token_extent=563 batch_extent=8
/// reduction_k=128`. `feature=576` is an exact `TILED_GEMM_BLOCK_M`(64)
/// multiple (every row-tile interior), `token=563` is NOT a
/// `TILED_GEMM_BLOCK_N`(32) multiple (`563 % 32 == 19`, the last column-tile
/// a genuine partial edge) -- a mixed interior/boundary grid at real
/// prefill scale.
#[test]
fn dense_batch_innermost_matches_restage_path_at_the_real_weather_prefill_shape() {
    check_dense_batch_innermost_byte_identity(563, 128, 576, 8, "weather_score_shape");
}

/// Real gemma4-E2B weather-prompt `attended` (P.V) shape, MEASURED via
/// `dstore/prompt5_strides_step0.log` (`node=162`): `feature_extent=256
/// token_extent=563 batch_extent=8 reduction_k=32`, `out_stride_feature=1`
/// -- this IS `DIRECT_STORE`-eligible in production (feature axis already
/// unit-stride, per `push_tiled_gemm_direct_store_arm`'s own gate). `256` is
/// an exact `TILED_GEMM_BLOCK_M`(64) multiple; `563 % TILED_GEMM_BLOCK_N(32)
/// == 19` gives a genuine partial edge column-tile in the same dispatch, so
/// this one shape exercises both the direct-store fast arm AND its
/// boundary-tile restage fallback at real prefill scale.
#[test]
fn dense_feature_fastest_matches_restage_path_at_the_real_weather_attended_shape() {
    check_dense_direct_store_byte_identity(563, 32, 8, 256, "weather_attended_shape");
}
