//! `PROXIMA_PACKED_LANE_RUN_BYTES` A/B switch on the packed matvec's lane map. Its own binary so the
//! env mutation below cannot race any other test's emit; every body runs under `temp_env::with_var`.
//! The key is read through `omega::plan(..).kernel_keys()`, the text from `omega::emit`; both are
//! public. Teaching pointers: the selection is `msl::packed_row_lane_run`, the render is
//! `FlatBlockForm::matvec_msl` under a `LaneRun`, the token is `MetalOnlyExtras::packed_row_lane_run`.
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

const LANE_RUN_ENV: &str = "PROXIMA_PACKED_LANE_RUN_BYTES";
const OUT_ROWS: u32 = 24;
const IN_DIM: u32 = 512;
const SCALAR_16_BIT_BYTES: usize = 2;
const RUN_MARKER: &str = "flat_block_levels";
const RUN_BODY: &str = "sumq * d";

/// the twenty pinned cases of `packed_row_codecs_msl_identity.rs`, by fixture name
const CASES: [(&str, Codec, bool); 20] = [
    ("q3k_plain", Codec::Q3K, false),
    ("q3k_negated", Codec::Q3K, true),
    ("q4k_plain", Codec::Q4K, false),
    ("q4k_negated", Codec::Q4K, true),
    ("q5k_plain", Codec::Q5K, false),
    ("q5k_negated", Codec::Q5K, true),
    ("q6k_plain", Codec::Q6K, false),
    ("q6k_negated", Codec::Q6K, true),
    ("q8_0_plain", Codec::Q8_0, false),
    ("q8_0_negated", Codec::Q8_0, true),
    ("q4_0_plain", Codec::Q4_0, false),
    ("q4_0_negated", Codec::Q4_0, true),
    ("fp6e3m2_plain", Codec::Fp6E3M2, false),
    ("fp6e3m2_negated", Codec::Fp6E3M2, true),
    ("iq4xs_plain", Codec::Iq4Xs, false),
    ("iq4xs_negated", Codec::Iq4Xs, true),
    ("iq3s_plain", Codec::Iq3S, false),
    ("iq3s_negated", Codec::Iq3S, true),
    ("f16_plain", Codec::Float16, false),
    ("bf16_plain", Codec::BFloat16, false),
];

struct Emitted {
    key: Option<String>,
    source: String,
}

fn matvec_program(negated: bool, rows: u32, reduction: u32) -> (Vec<Op>, NodeId) {
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
    let reduced = if negated {
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Negate,
                operands: vec![(product, IndexMap::Affine(projection(3, &[0, 1, 2])))],
                name: None,
            },
        )
    } else {
        product
    };
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: reduced,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

fn plan_kernel_keys(program: &[Op], sum: NodeId, codec: Codec, rows: u32, reduction: u32, policy: NumericPolicy) -> Option<Vec<String>> {
    let (elements, block_bytes) = match codec.block_layout() {
        Some(layout) => (layout.elements as usize, layout.block_bytes as usize),
        None => (1, SCALAR_16_BIT_BYTES),
    };
    let weight_bytes = vec![0u8; rows as usize * (reduction as usize / elements) * block_bytes];
    let activation = vec![0.5f32; reduction as usize];
    let blocks = [QuantizedBlock::Packed { codec, bytes: &weight_bytes }, QuantizedBlock::Float32(&activation)];
    let plan = omega::plan(program, &[], &blocks, &[sum], policy).ok()?;
    plan.kernel_keys().ok()
}

fn emit_op(codec: Codec, negated: bool, rows: u32, reduction: u32, policy: NumericPolicy) -> Emitted {
    let (program, sum) = matvec_program(negated, rows, reduction);
    let shapes = proxima_tensor::infer(&program, &[]).expect("the synthetic program infers");
    let packed: omega::PackedOperands = [(NodeId(0), codec)].into_iter().collect();
    let mut bound = proxima_tensor::bind(&program, &shapes, &[sum], policy).expect("binds");
    proxima_tensor::correct_packed_matmul_layouts(&mut bound, &[NodeId(0)].into_iter().collect());
    let resolved = bound.iter().find(|operation| operation.node == sum).expect("the reduce node is bound");
    let kernel = omega::emit(resolved, &packed, policy).expect("emits");
    let key = plan_kernel_keys(&program, sum, codec, rows, reduction, policy).map(|keys| keys.join(" "));
    Emitted { key, source: kernel.source }
}

impl Emitted {
    /// the key of a shape `omega::plan` accepts; a codec whose block does not tile the synthetic
    /// shape has no plan, so its identity is asserted through the source alone
    fn key(&self) -> &str {
        self.key.as_deref().expect("the shape plans, so it has a kernel key")
    }
}

fn fixture(file_name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/packed_row_codecs").join(file_name);
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("fixture {} is readable: {error}", path.display()))
}

#[proxima::test]
#[case::env_unset(None)]
#[case::env_zero(Some("0"))]
#[case::env_empty(Some(""))]
async fn every_pinned_identity_case_is_byte_identical_and_carries_no_lane_run_token_while_the_switch_is_off(#[case] value: Option<&str>) {
    temp_env::with_var(LANE_RUN_ENV, value, || {
        let mut keyed = 0;
        for (name, codec, negated) in CASES {
            let emitted = emit_op(codec, negated, OUT_ROWS, IN_DIM, NumericPolicy::default());

            assert!(emitted.source == fixture(&format!("{name}.msl")), "{name}: source drifted from its pinned fixture with {LANE_RUN_ENV}={value:?}");
            if let Some(key) = &emitted.key {
                keyed += 1;
                assert!(!key.contains("_lr"), "{name}: {key}");
            }
        }
        assert!(keyed > 0, "no case produced a kernel key, so the token check ran zero times");
    });
}

#[test]
fn env_eight_under_llama_relaxed_renders_the_lane_run_and_keys_it_apart_from_the_fixture() {
    temp_env::with_var(LANE_RUN_ENV, Some("8"), || {
        let emitted = emit_op(Codec::Q8_0, false, OUT_ROWS, IN_DIM, NumericPolicy::llama_relaxed());

        assert!(emitted.key().contains("_lr8"), "{}", emitted.key());
        assert!(emitted.source.contains(RUN_MARKER), "{}", emitted.source);
        assert!(emitted.source.contains(RUN_BODY), "{}", emitted.source);
        assert!(emitted.source != fixture("q8_0_plain.msl"), "the lane-run body must differ from the lane-per-block fixture");
    });
}

#[test]
fn env_eight_under_the_default_bit_exact_policy_is_refused_and_keeps_the_fixture_text() {
    temp_env::with_var(LANE_RUN_ENV, Some("8"), || {
        let emitted = emit_op(Codec::Q8_0, false, OUT_ROWS, IN_DIM, NumericPolicy::default());

        assert!(!emitted.key().contains("_lr"), "{}", emitted.key());
        assert!(!emitted.source.contains(RUN_BODY), "{}", emitted.source);
        assert!(emitted.source == fixture("q8_0_plain.msl"), "a refused lane run must leave the pinned q8_0_plain text");
    });
}

#[test]
fn env_three_does_not_divide_the_plane_so_the_sized_default_keeps_the_lane_per_block_text() {
    let unset = temp_env::with_var(LANE_RUN_ENV, None::<&str>, || emit_op(Codec::Q8_0, false, OUT_ROWS, IN_DIM, NumericPolicy::llama_relaxed()));
    temp_env::with_var(LANE_RUN_ENV, Some("3"), || {
        let emitted = emit_op(Codec::Q8_0, false, OUT_ROWS, IN_DIM, NumericPolicy::llama_relaxed());

        assert!(!emitted.key().contains("_lr"), "{}", emitted.key());
        assert!(emitted.source == unset.source, "env 3 is out of range and must read as the sized default (off)");
        assert!(emitted.key == unset.key, "{:?} vs {:?}", emitted.key, unset.key);
    });
}

#[test]
fn a_layout_without_a_whole_byte_code_read_takes_no_lane_run_even_when_the_policy_admits_it() {
    temp_env::with_var(LANE_RUN_ENV, Some("8"), || {
        for (name, codec) in [("q4k", Codec::Q4K), ("q6k", Codec::Q6K), ("f16", Codec::Float16)] {
            let emitted = emit_op(codec, false, OUT_ROWS, IN_DIM, NumericPolicy::llama_relaxed());

            assert!(!emitted.key().contains("_lr"), "{name}: {}", emitted.key());
            assert!(!emitted.source.contains(RUN_BODY), "{name}");
        }
    });
}
