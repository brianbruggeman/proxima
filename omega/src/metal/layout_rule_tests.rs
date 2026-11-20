use proxima_numeric::codec::Codec;
use proxima_numeric::float_layout::FloatLayout;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::spec::{elementwise, input_leaf, reduce};
use proxima_tensor::{Extent, NodeId, Op, QuantizedBlock, ReduceInit, ScalarOp};

use super::*;

const TOKENS: usize = 512;
const REDUCTION: usize = 2048;
const FEATURES: usize = 256;

fn router_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![
            Extent::Static(FEATURES as u32),
            Extent::Static(REDUCTION as u32),
        ],
        "weight",
    );
    let activation = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![
            Extent::Static(TOKENS as u32),
            Extent::Static(REDUCTION as u32),
        ],
        "activation",
    );
    let product = elementwise(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Multiply,
        &[(weight, "fi->tfi"), (activation, "ti->tfi")],
    )
    .expect("router product has compatible axes");
    let output = reduce(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "tfi->tfi",
        "tf->tfi",
    )
    .expect("router reduction keeps token and feature axes");
    (program, output)
}

fn router_plan_source(layout: Option<DenseMmaInputLayout>) -> (String, Option<FloatLayout>) {
    let (program, output) = router_program();
    let weight_bytes = vec![0u8; FEATURES * REDUCTION * 2];
    let activation = vec![0.5f32; TOKENS * REDUCTION];
    let blocks = [
        QuantizedBlock::Packed {
            codec: Codec::BFloat16,
            bytes: &weight_bytes,
        },
        QuantizedBlock::Float32(&activation),
    ];
    let serving = NumericPolicy::llama_relaxed().with_epilogue_sources(true);
    let plan = plan_with_placed_inputs(
        &program,
        &[],
        &blocks,
        &[output],
        (serving, layout),
        &[],
        (true, false),
    )
    .expect("the router plans");
    let bound = plan.prepared.resolved.first().expect("one fused router op");
    let source = crate::msl::emit(bound, &plan.packed_operands, plan.numeric_policy)
        .expect("router emits")
        .source;
    (source, plan.dense_mma_input_layout)
}

#[test]
fn an_explicit_bf16_layout_reaches_the_serving_plan_as_bf16_tiles() {
    let (source, fixed) =
        router_plan_source(Some(DenseMmaInputLayout::explicit(FloatLayout::BF16)));
    assert!(
        source.contains("simdgroup_bfloat8x8"),
        "bf16 layout grants bf16 tiles"
    );
    assert_eq!(fixed, Some(FloatLayout::BF16));
}

#[test]
fn an_unconfigured_layout_keeps_the_float_tiles_and_fixes_no_plan_layout() {
    let (source, fixed) = router_plan_source(None);
    assert!(source.contains("simdgroup_float8x8") && !source.contains("simdgroup_bfloat8x8"));
    assert_eq!(fixed, None);
}

#[cfg(feature = "metal-output-placement")]
fn named_router_source_under_env_layout(layout: &str) -> String {
    let runtime = temp_env::with_vars(
        [("OMEGA_DENSE_MMA_INPUT_LAYOUT", Some(layout))],
        RuntimeConfig::from_env,
    )
    .expect("the layout value parses from the environment");
    let (program, output) = router_program();
    let weight_bytes = vec![0u8; FEATURES * REDUCTION * 2];
    let activation = vec![0.5f32; TOKENS * REDUCTION];
    let named = [
        (
            "weight",
            QuantizedBlock::Packed {
                codec: Codec::BFloat16,
                bytes: &weight_bytes,
            },
        ),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let serving = NumericPolicy::llama_relaxed().with_epilogue_sources(true);
    let plan = plan_named_with_placed_inputs_laid_out(
        &program,
        &[],
        &named,
        &[output],
        (serving, Some(runtime.dense_mma_input_layout)),
        &[],
        (true, false),
    )
    .expect("the router plans through the serving entry point");
    let bound = plan.prepared.resolved.first().expect("one fused router op");
    crate::msl::emit(bound, &plan.packed_operands, plan.numeric_policy)
        .expect("router emits")
        .source
}

#[cfg(feature = "metal-output-placement")]
#[test]
fn the_serving_entry_renders_bf16_tiles_for_a_runtime_config_layout_of_bf16() {
    let source = named_router_source_under_env_layout("bf16");
    assert!(source.contains("simdgroup_bfloat8x8"));
}

#[cfg(feature = "metal-output-placement")]
#[test]
fn the_serving_entry_renders_float_tiles_for_a_runtime_config_layout_of_f32() {
    let source = named_router_source_under_env_layout("f32");
    assert!(source.contains("simdgroup_float8x8") && !source.contains("simdgroup_bfloat8x8"));
}
