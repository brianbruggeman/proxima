//! `PROXIMA_PACKED_K_SPLIT` / `PROXIMA_PACKED_ROWS_KSPLIT` A/B switch on the packed matvec geometry.
//! Its own binary so the env mutation below cannot race any other test's emit; every body runs under
//! `temp_env::with_vars`. The key is read through `omega::plan(..).kernel_keys()`, the width is
//! `Kernel::grid.threadgroup_width` from `omega::emit`; both are public.
#![cfg(all(
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-q4k-mask-fma")
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use omega::Codec;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, projection};

const K_SPLIT_ENV: &str = "PROXIMA_PACKED_K_SPLIT";
const ROWS_ENV: &str = "PROXIMA_PACKED_ROWS_KSPLIT";

const ATTN_Q_ROWS: u32 = 4096;
const ATTN_Q_K: u32 = 2048;
const ATTN_OUTPUT_ROWS: u32 = 2048;
const ATTN_OUTPUT_K: u32 = 4096;
const SCALAR_16_BIT_BYTES: usize = 2;
const ROUTER_ROWS: u32 = 256;
const ROUTER_K: u32 = 2048;

struct Emitted {
    key: String,
    kernel: omega::Kernel,
}

fn matvec_program(rows: u32, reduction: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(&mut program, Op::Input { dtype: DType::UINT8, shape: vec![Extent::Static(rows), Extent::Static(reduction)], name: None });
    let activation = append(&mut program, Op::Input { dtype: DType::FLOAT32, shape: vec![Extent::Static(1), Extent::Static(reduction)], name: None });
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
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
            dtype: DType::FLOAT32,
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

fn emit_op(codec: Codec, rows: u32, reduction: u32) -> Emitted {
    let (program, sum) = matvec_program(rows, reduction);
    let shapes = proxima_tensor::infer(&program, &[]).expect("the synthetic program infers");
    let packed: omega::PackedOperands = [(NodeId(0), codec)].into_iter().collect();
    let mut bound = proxima_tensor::bind(&program, &shapes, &[sum], NumericPolicy::default()).expect("binds");
    proxima_tensor::correct_packed_matmul_layouts(&mut bound, &[NodeId(0)].into_iter().collect());
    let resolved = bound.iter().find(|op| op.node == sum).expect("the reduce node is bound");
    let kernel = omega::emit(resolved, &packed, NumericPolicy::default()).expect("emits");
    let key = plan_kernel_keys(&program, sum, codec, rows, reduction).join(" ");
    Emitted { key, kernel }
}

/// The kernel keys `omega::plan` compiles under, read through `MetalPlan::kernel_keys` as
/// `bf16_row_blocked_matvec_parity.rs` does. The emitted entry name has no `_ks` token: only the
/// cache key carries it (`identity.rs` appends it, `signature_tokens_prelude.rs`'s Reduce arm does not).
fn plan_kernel_keys(program: &[Op], sum: NodeId, codec: Codec, rows: u32, reduction: u32) -> Vec<String> {
    let (elements, block_bytes) = match codec.block_layout() {
        Some(layout) => (layout.elements as usize, layout.block_bytes as usize),
        None => (1, SCALAR_16_BIT_BYTES),
    };
    let weight_bytes = vec![0u8; rows as usize * (reduction as usize / elements) * block_bytes];
    let activation = vec![0.5f32; reduction as usize];
    let blocks = [QuantizedBlock::Packed { codec, bytes: &weight_bytes }, QuantizedBlock::Float32(&activation)];
    let plan = omega::plan(program, &[], &blocks, &[sum], NumericPolicy::default()).expect("metal plans the matvec");
    plan.kernel_keys().expect("the plan exposes its kernel keys")
}

fn fixture(file_name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/packed_row_codecs").join(file_name);
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("fixture {} is readable: {error}", path.display()))
}

#[test]
fn env_unset_leaves_the_attn_q_q8_0_op_unsplit_and_byte_identical_to_the_pinned_fixture() {
    temp_env::with_vars([(K_SPLIT_ENV, None::<&str>), (ROWS_ENV, None)], || {
        let emitted = emit_op(Codec::Q8_0, ATTN_Q_ROWS, ATTN_Q_K);

        assert!(!emitted.key.contains("_ks"), "{}", emitted.key);
        assert!(emitted.kernel.source == fixture("q8_0_plain.msl"), "unset env drifted from the pinned q8_0_plain fixture");
    });
}

#[proxima::test]
#[case::split_2_on_attn_q(ATTN_Q_ROWS, ATTN_Q_K, "2")]
#[case::split_4_on_attn_q(ATTN_Q_ROWS, ATTN_Q_K, "4")]
#[case::split_4_on_attn_output(ATTN_OUTPUT_ROWS, ATTN_OUTPUT_K, "4")]
async fn a_whole_block_lane_q8_0_op_never_splits_under_the_env(#[case] rows: u32, #[case] reduction: u32, #[case] split: &str) {
    temp_env::with_vars([(K_SPLIT_ENV, Some(split)), (ROWS_ENV, None)], || {
        let emitted = emit_op(Codec::Q8_0, rows, reduction);

        assert!(!emitted.key.contains("_ks"), "{}", emitted.key);
        assert!(emitted.kernel.source == fixture("q8_0_plain.msl") || rows != ATTN_Q_ROWS, "split {split} changed the Q8_0 attn_q kernel");
        assert!(!emitted.kernel.source.contains("partial_sums"), "{}", emitted.kernel.source);
    });
}

#[proxima::test]
#[case::split_2_is_64_wide("2", "_ks2", 64)]
#[case::split_4_is_128_wide("4", "_ks4", 128)]
async fn k_split_on_the_bf16_router_splits_simdgroups_over_a_threadgroup_of_that_many_warps(#[case] split: &str, #[case] token: &str, #[case] width: u64) {
    temp_env::with_vars([(K_SPLIT_ENV, Some(split)), (ROWS_ENV, None)], || {
        let emitted = emit_op(Codec::BFloat16, ROUTER_ROWS, ROUTER_K);

        assert!(emitted.key.contains(token), "{}", emitted.key);
        assert!(emitted.kernel.source.contains("sgitg"), "{}", emitted.kernel.source);
        assert_eq!(emitted.kernel.grid.threadgroup_width, Some(width));
    });
}

#[test]
fn rows_ksplit_4_with_a_split_of_2_folds_four_rows_per_simdgroup_of_the_bf16_router() {
    temp_env::with_vars([(K_SPLIT_ENV, Some("2")), (ROWS_ENV, Some("4"))], || {
        let emitted = emit_op(Codec::BFloat16, ROUTER_ROWS, ROUTER_K);

        assert!(emitted.key.contains("_ks2"), "{}", emitted.key);
        assert!(emitted.kernel.source.contains("partial_sums[2][4]"), "{}", emitted.kernel.source);
    });
}

#[test]
fn env_unset_splits_the_bf16_router_four_ways_by_default() {
    temp_env::with_vars([(K_SPLIT_ENV, None::<&str>), (ROWS_ENV, None)], || {
        let emitted = emit_op(Codec::BFloat16, ROUTER_ROWS, ROUTER_K);

        assert!(emitted.key.contains("_ks4"), "{}", emitted.key);
        assert_eq!(emitted.kernel.grid.threadgroup_width, Some(128));
    });
}

#[test]
fn env_split_1_leaves_the_bf16_router_unsplit() {
    temp_env::with_vars([(K_SPLIT_ENV, Some("1")), (ROWS_ENV, None)], || {
        let emitted = emit_op(Codec::BFloat16, ROUTER_ROWS, ROUTER_K);

        assert!(!emitted.key.contains("_ks"), "{}", emitted.key);
    });
}

#[proxima::test]
#[case::zero_is_below_the_range("0")]
#[case::nine_is_above_the_range("9")]
#[case::text_does_not_parse("x")]
#[case::empty_does_not_parse("")]
async fn an_out_of_range_or_unparsable_k_split_keeps_the_default(#[case] value: &str) {
    temp_env::with_vars([(K_SPLIT_ENV, Some(value)), (ROWS_ENV, None)], || {
        let emitted = emit_op(Codec::Q8_0, ATTN_Q_ROWS, ATTN_Q_K);

        assert!(!emitted.key.contains("_ks"), "{value:?}: {}", emitted.key);
        assert!(emitted.kernel.source == fixture("q8_0_plain.msl"), "{value:?} changed the emitted kernel");
    });
}

#[test]
fn a_sub_block_q5_k_body_never_splits_under_the_env() {
    temp_env::with_vars([(K_SPLIT_ENV, Some("4")), (ROWS_ENV, None)], || {
        let emitted = emit_op(Codec::Q5K, ATTN_OUTPUT_ROWS, ATTN_OUTPUT_K);

        assert!(!emitted.key.contains("_ks"), "{}", emitted.key);
    });
}
