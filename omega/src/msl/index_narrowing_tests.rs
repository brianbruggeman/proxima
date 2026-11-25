use alloc::collections::BTreeMap;

#[cfg(feature = "metal-grouped-gemm")]
use crate::epilogue::index32_reach_fits;

#[cfg(feature = "metal-attn-variants")]
use proxima_numeric::float_layout::FloatLayout;
use super::tests::embedding_lookup_op;
#[cfg(feature = "metal-grouped-gemm")]
use super::tests::gathered_matmul_op;
use super::*;

const NORM_BASE: &str = include_str!("../../tests/fixtures/index32/norm_epilogue_1536.base.msl");
const NORM_TIP: &str = include_str!("../../tests/fixtures/index32/norm_epilogue_1536.tip.msl");
const COMBINE_BASE: &str =
    include_str!("../../tests/fixtures/index32/combine_epilogue_1024.base.msl");
const COMBINE_TIP: &str =
    include_str!("../../tests/fixtures/index32/combine_epilogue_1024.tip.msl");
const EXPERT_BASE: &str = include_str!("../../tests/fixtures/index32/expert_down_q8_0.base.msl");
const EXPERT_TIP: &str = include_str!("../../tests/fixtures/index32/expert_down_q8_0.tip.msl");
const EMBEDDING_BASE: &str =
    include_str!("../../tests/fixtures/index32/embedding_q6k_1536.base.msl");
const EMBEDDING_TIP: &str = include_str!("../../tests/fixtures/index32/embedding_q6k_1536.tip.msl");
const LOOKUP_BASE: &str =
    include_str!("../../tests/fixtures/index32/embedding_lookup_f32.base.msl");
const LOOKUP_TIP: &str = include_str!("../../tests/fixtures/index32/embedding_lookup_f32.tip.msl");

#[cfg(feature = "metal-grouped-gemm")]
const GROUPED_BASE: &str =
    include_str!("../../tests/fixtures/index32/grouped_gemm_down_q8_0.base.msl");
#[cfg(feature = "metal-grouped-gemm")]
const GROUPED_TIP: &str =
    include_str!("../../tests/fixtures/index32/grouped_gemm_down_q8_0.tip.msl");
#[cfg(feature = "metal-tiled-gemm")]
const DENSE_BASE: &str =
    include_str!("../../tests/fixtures/index32/dense_batched_gemm_score.base.msl");
#[cfg(feature = "metal-tiled-gemm")]
const DENSE_TIP: &str =
    include_str!("../../tests/fixtures/index32/dense_batched_gemm_score.tip.msl");

fn kernel_body(source: &str) -> &str {
    &source[source.find("kernel void ").expect("a kernel entry")..]
}

fn has_word(text: &str, word: &str) -> bool {
    replace_word(text, word, "") != text
}

fn emitted(
    bound: &BoundOp,
    packed: &BTreeMap<NodeId, Codec>,
    coord_index32: bool,
) -> String {
    let knobs = EmitKnobs::default().with_coord_index32(coord_index32);
    emit_with_knobs(bound, packed, NumericPolicy::default(), &knobs)
        .expect("kernel emits")
        .source
}

#[cfg(feature = "metal-tiled-gemm")]
fn dense_knobs(coord_index32: bool) -> EmitKnobs {
    EmitKnobs::default()
        .with_tiled_gemm_dense(true)
        .with_coord_index32(coord_index32)
}

#[cfg(feature = "metal-tiled-gemm")]
fn emitted_dense(bound: &BoundOp, coord_index32: bool) -> String {
    emit_with_knobs(
        bound,
        &BTreeMap::new(),
        NumericPolicy::llama_relaxed(),
        &dense_knobs(coord_index32),
    )
    .expect("dense tiled kernel emits")
    .source
}

#[cfg(feature = "metal-tiled-gemm")]
fn dense_cache_key(bound: &BoundOp) -> String {
    kernel_cache_key_with_knobs(
        bound,
        &BTreeMap::new(),
        NumericPolicy::llama_relaxed(),
        &dense_knobs(true),
    )
    .expect("cache key builds")
}

fn cache_key(bound: &BoundOp, packed: &BTreeMap<NodeId, Codec>) -> String {
    kernel_cache_key_with_knobs(bound, packed, NumericPolicy::default(), &EmitKnobs::default())
        .expect("cache key builds")
}

fn packed_table(bound: &BoundOp, codec: Codec) -> BTreeMap<NodeId, Codec> {
    BTreeMap::from([(bound.operands()[0].0, codec)])
}

#[test]
fn the_captured_hidden_width_norm_narrows_to_the_recorded_32_bit_kernel() {
    assert_eq!(narrow_body_to_int32(NORM_BASE), NORM_TIP);
}

#[test]
fn the_captured_combine_fold_narrows_to_the_recorded_32_bit_kernel() {
    assert_eq!(narrow_body_to_int32(COMBINE_BASE), COMBINE_TIP);
}

#[test]
fn the_captured_expert_projection_narrows_only_its_two_prologue_divides() {
    let narrowed = narrow_expert_prologue(EXPERT_BASE);

    assert_eq!(narrowed, EXPERT_TIP);
    assert_eq!(EXPERT_BASE.lines().count(), narrowed.lines().count());
    let changed = EXPERT_BASE
        .lines()
        .zip(narrowed.lines())
        .filter(|(base, tip)| base != tip)
        .count();
    assert_eq!(changed, 2);
}

#[test]
fn the_captured_gathered_embedding_decodes_its_block_index_in_unsigned_32_bit() {
    let narrowed = narrow_block_remainders(&narrow_block_decode(EMBEDDING_BASE));

    assert_eq!(narrowed, EMBEDDING_TIP);
    assert!(
        narrowed
            .contains("q6k_element(in0 + (long)((uint)off0 / 256u) * 210, ((uint)off0 % 256u))")
    );
}

#[test]
fn the_captured_f32_embedding_lookup_narrows_to_the_recorded_32_bit_kernel() {
    assert_eq!(narrow_body_to_int32(LOOKUP_BASE), LOOKUP_TIP);
}

#[test]
fn the_uniform_struct_keeps_its_64_bit_fields_when_the_body_narrows() {
    let narrowed = narrow_body_to_int32(NORM_BASE);

    assert!(narrowed.contains("    long output_total;\n    long reduction_total;"));
    assert!(!has_word(kernel_body(&narrowed), "long"));
    assert!(has_word(kernel_body(NORM_BASE), "long"));
}

#[test]
fn the_body_pass_wraps_nested_uniform_reads_and_drops_the_64_bit_literal_suffix() {
    let source = "struct Uniforms {\n    long operand_strides[2][2];\n    long reduction_total;\n};\n\
                  kernel void entry() {\n    long span = u.operand_strides[0][(1)] * 32L + (long)ulong_value;\n    ulong keep = 7UL;\n    long total = u.reduction_total;\n}\n";

    let narrowed = narrow_body_to_int32(source);

    assert!(
        narrowed.contains("int span = ((int)u.operand_strides[0][(1)]) * 32 + (int)ulong_value;"),
        "{narrowed}"
    );
    assert!(narrowed.contains("ulong keep = 7UL;"), "{narrowed}");
    assert!(
        narrowed.contains("int total = ((int)u.reduction_total);"),
        "{narrowed}"
    );
}

#[test]
fn a_block_read_that_is_not_an_offset_decode_is_left_alone() {
    let source = "float value = in0[(off0 + 3)] + q4_0_element(in1 + (index / 32) * 18, (uint)(index % 32));";

    assert_eq!(
        narrow_block_remainders(&narrow_block_decode(source)),
        source
    );
}

#[cfg(all(
    feature = "reduce-epilogue-fusion",
    feature = "metal-wide-cooperative-reduce"
))]
mod norm_rows {
    use super::super::cooperative_fold_tests::normalization_rows::rmsnorm_over_row;
    use super::*;

    const NO_PACKED: BTreeMap<NodeId, Codec> = BTreeMap::new();

    #[test]
    fn a_hidden_width_norm_row_renders_its_index_arithmetic_in_32_bit_under_the_c32_key() {
        let bound = rmsnorm_over_row(1536);

        let source = emitted(&bound, &NO_PACKED, true);

        assert!(!has_word(kernel_body(&source), "long"), "{source}");
        assert!(
            source.contains("int epi_pre_r = (int)lane + (int)slot * "),
            "{source}"
        );
        assert!(
            source.contains("= ((int)u.epilogue_operand_base["),
            "{source}"
        );
        assert!(source.contains("    long output_total;"));
        assert!(cache_key(&bound, &NO_PACKED).contains("_c32"));
    }

    #[test]
    fn the_coordinate_override_set_to_zero_restores_the_64_bit_norm_body() {
        let bound = rmsnorm_over_row(1536);

        let source = emitted(&bound, &NO_PACKED, false);

        assert!(
            source.contains("long epi_pre_r = (long)lane + (long)slot * "),
            "{source}"
        );
        assert!(!source.contains("int epi_pre_r"));
    }

    #[test]
    fn a_norm_row_exactly_at_the_reach_limit_still_narrows() {
        let bound = rmsnorm_over_row(1 << 30);

        assert!(cache_key(&bound, &NO_PACKED).contains("_c32"));
        assert!(!has_word(
            kernel_body(&emitted(&bound, &NO_PACKED, true)),
            "long"
        ));
    }

    #[test]
    fn a_norm_row_one_past_the_reach_limit_keeps_the_64_bit_body_and_drops_the_key_bit() {
        let bound = rmsnorm_over_row((1 << 30) + 1);

        let source = emitted(&bound, &NO_PACKED, true);

        assert!(!cache_key(&bound, &NO_PACKED).contains("_c32"));
        assert!(
            source.contains("long epi_pre_r = (long)lane + (long)slot * "),
            "{source}"
        );
        assert!(!source.contains("int epi_pre_r"));
    }
}

#[test]
fn a_gathered_f32_lookup_renders_its_offsets_and_fault_check_in_32_bit() {
    let bound = embedding_lookup_op(50_000, 8, 4);

    let source = emitted(&bound, &BTreeMap::new(), true);

    assert!(!has_word(kernel_body(&source), "long"), "{source}");
    assert!(
        source.contains("fetched0 < 0 || fetched0 >= ((int)u.gather_extent[0])"),
        "{source}"
    );
    assert!(cache_key(&bound, &BTreeMap::new()).contains("_c32"));
}

#[test]
fn a_q6_k_embedding_table_past_i32_but_inside_u32_decodes_its_block_in_unsigned_32_bit() {
    let bound = embedding_lookup_op(262_144, 8_960, 1);
    let packed = packed_table(&bound, Codec::Q6K);

    let source = emitted(&bound, &packed, true);

    assert!(
        source.contains("(long)((uint)off0 / 256u) * 210"),
        "{source}"
    );
    assert!(source.contains("((uint)off0 % 256u)"), "{source}");
    assert!(!source.contains("(off0 / 256)"));
    assert!(cache_key(&bound, &packed).contains("_c32"));
}

#[test]
fn a_q6_k_embedding_table_past_u32_keeps_the_64_bit_block_decode() {
    let bound = embedding_lookup_op(524_288, 8_960, 1);
    let packed = packed_table(&bound, Codec::Q6K);

    let source = emitted(&bound, &packed, true);

    assert!(source.contains("(off0 / 256) * 210"), "{source}");
    assert!(source.contains("(uint)(off0 % 256)"), "{source}");
    assert!(!cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-grouped-gemm")]
#[test]
fn a_routed_q8_0_expert_projection_divides_its_flat_row_in_unsigned_32_bit() {
    let bound = gathered_matmul_op(8, 32, 512, 1024);
    let packed = packed_table(&bound, Codec::Q8_0);

    let source = emitted(&bound, &packed, true);

    assert!(
        source.contains("long expert_slot = (long)((uint)group_first / (uint)group_out_extent);"),
        "{source}"
    );
    assert!(
        source.contains("long out_row_base = (long)((uint)group_first % (uint)group_out_extent);"),
        "{source}"
    );
    assert!(cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-grouped-gemm")]
#[test]
fn a_routed_expert_projection_one_row_group_inside_u32_still_narrows() {
    let bound = gathered_matmul_op(63, 2, 67_108_864, 1024);
    let packed = packed_table(&bound, Codec::Q8_0);

    let source = emitted(&bound, &packed, true);

    assert!(
        source.contains("(uint)group_first / (uint)group_out_extent"),
        "{source}"
    );
    assert!(cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-grouped-gemm")]
#[test]
fn a_routed_expert_projection_with_outputs_past_u32_keeps_the_64_bit_divides() {
    let tokens = 63;
    assert!(
        u64::from(tokens) < crate::sized::TILED_GEMM_MIN_TOKENS,
        "this test pins the row-blocked body, so its token count must sit under the tiled minimum"
    );
    let rows = (u32::MAX / tokens + 1).next_multiple_of(8);
    let bound = gathered_matmul_op(tokens, 2, rows, 1024);
    let packed = packed_table(&bound, Codec::Q8_0);

    let source = emitted(&bound, &packed, true);

    assert!(
        source.contains("long expert_slot = group_first / group_out_extent;"),
        "{source}"
    );
    assert!(
        source.contains("long out_row_base = group_first % group_out_extent;"),
        "{source}"
    );
    assert!(!cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-grouped-gemm")]
#[test]
fn the_coordinate_override_set_to_zero_restores_every_family_to_the_64_bit_form() {
    let lookup = embedding_lookup_op(50_000, 8, 4);
    let embedding = embedding_lookup_op(262_144, 8_960, 1);
    let expert = gathered_matmul_op(8, 32, 512, 1024);

    let lookup_source = emitted(&lookup, &BTreeMap::new(), false);
    let embedding_source = emitted(&embedding, &packed_table(&embedding, Codec::Q6K), false);
    let expert_source = emitted(&expert, &packed_table(&expert, Codec::Q8_0), false);

    assert!(has_word(kernel_body(&lookup_source), "long"));
    assert!(embedding_source.contains("(off0 / 256) * 210"));
    assert!(expert_source.contains("long expert_slot = group_first / group_out_extent;"));
}

#[cfg(feature = "metal-grouped-gemm")]
fn first_kernel(source: &str) -> &str {
    let body = kernel_body(source);
    &body[..body[1..].find("kernel void ").map_or(body.len(), |at| at + 1)]
}

#[cfg(feature = "metal-grouped-gemm")]
fn flint_down(experts: u32, rows: u32, k: u32) -> (BoundOp, BTreeMap<NodeId, Codec>) {
    let bound = super::tests::stacked_gathered_matmul_op(512, 6, experts, rows, k, true);
    let packed = packed_table(&bound, Codec::Q8_0);
    (bound, packed)
}

#[cfg(feature = "metal-grouped-gemm")]
#[test]
fn the_captured_grouped_down_projection_narrows_to_the_recorded_32_bit_kernel() {
    assert_eq!(narrow_tiled_gemm(GROUPED_BASE), GROUPED_TIP);
}

#[cfg(feature = "metal-grouped-gemm")]
#[test]
fn a_grouped_down_projection_renders_the_recorded_fixtures_under_each_coordinate_setting() {
    let (bound, packed) = flint_down(256, 2048, 512);

    assert_eq!(emitted(&bound, &packed, false), GROUPED_BASE);
    assert_eq!(emitted(&bound, &packed, true), GROUPED_TIP);
    assert!(cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-grouped-gemm")]
#[test]
fn the_grouped_gemm_kernel_drops_every_long_but_its_route_prepass_keeps_them() {
    assert!(!has_word(first_kernel(GROUPED_TIP), "long"));
    assert!(has_word(first_kernel(GROUPED_BASE), "long"));
    let prepass = |source: &'static str| &source[source.find("_route_prepass(").expect("prepass kernel")..];
    assert_eq!(prepass(GROUPED_TIP), prepass(GROUPED_BASE));
    assert!(has_word(prepass(GROUPED_TIP), "long"));
}

#[cfg(feature = "metal-grouped-gemm")]
#[test]
fn a_grouped_weight_stack_past_the_reach_limit_keeps_the_64_bit_kernel() {
    let (bound, packed) = flint_down(256, 2048, 4096);

    let source = emitted(&bound, &packed, true);

    assert_eq!(source, emitted(&bound, &packed, false));
    assert!(source.contains("long w_elem = u.operand_base[0]"), "{source}");
    assert!(!cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-grouped-gemm")]
#[test]
fn a_grouped_weight_stack_whose_elements_fit_but_whose_q8_0_bytes_do_not_keeps_long() {
    let (bound, packed) = flint_down(256, 2000, 2048);
    assert!(
        gathered_reach(
            &bound.operands()[0].1,
            bound.operands()[0].2.as_ref(),
            &bound.extents
        )
        .is_some_and(index32_reach_fits),
        "the element reach must fit for this test to isolate the byte reach"
    );

    assert!(!cache_key(&bound, &packed).contains("_c32"));
    assert!(emitted(&bound, &packed, true).contains("long w_elem"));
}

#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn the_captured_dense_batched_gemm_narrows_to_the_recorded_32_bit_kernel() {
    assert_eq!(narrow_tiled_gemm(DENSE_BASE), DENSE_TIP);
}

#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn a_dense_batched_gemm_renders_the_recorded_fixtures_under_each_coordinate_setting() {
    let bound = super::tests::dense_batched_score_shaped_op(563, 576, 8, 128);

    assert_eq!(emitted_dense(&bound, false), DENSE_BASE);
    assert_eq!(emitted_dense(&bound, true), DENSE_TIP);
    assert!(dense_cache_key(&bound).contains("_c32"));
    assert!(!has_word(kernel_body(DENSE_TIP), "long"));
}

#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn a_dense_batched_gemm_whose_operand_reach_passes_the_limit_keeps_the_64_bit_kernel() {
    let bound = super::tests::dense_batched_score_shaped_op(563, 576, 8, 1 << 20);

    let source = emitted_dense(&bound, true);

    assert_eq!(source, emitted_dense(&bound, false));
    assert!(source.contains("long aoff = u.operand_base[1]"), "{source}");
    assert!(!dense_cache_key(&bound).contains("_c32"));
}

#[test]
fn the_body_pass_drops_a_lowercase_64_bit_literal_suffix_but_not_an_identifier_ending_in_l() {
    let source = "struct Uniforms {\n    long n;\n};\nkernel void entry() {\n    long tiles = (u.n + 63l) / 64l;\n    long call1 = 2;\n}\n";

    let narrowed = narrow_body_to_int32(source);

    assert!(
        narrowed.contains("int tiles = (((int)u.n) + 63) / 64;"),
        "{narrowed}"
    );
    assert!(narrowed.contains("int call1 = 2;"), "{narrowed}");
}

#[cfg(feature = "metal-attn-split-rows")]
const FULL_LAYER_BASE: &str =
    include_str!("../../tests/fixtures/index32/cached_attention_full_layer.base.msl");
#[cfg(feature = "metal-attn-split-rows")]
const FULL_LAYER_TIP: &str =
    include_str!("../../tests/fixtures/index32/cached_attention_full_layer.tip.msl");
#[cfg(feature = "metal-attn-split-rows")]
const SLIDING_LAYER_BASE: &str =
    include_str!("../../tests/fixtures/index32/cached_attention_sliding_layer.base.msl");
#[cfg(feature = "metal-attn-split-rows")]
const SLIDING_LAYER_TIP: &str =
    include_str!("../../tests/fixtures/index32/cached_attention_sliding_layer.tip.msl");

/// the shape of one flint prefill chunk's attention: `query_rows` new rows over
/// a `cached_key_rows` cache, f16 K and V, a sink, and the llama-relaxed policy
/// that selects the row-tiled form.
#[cfg(feature = "metal-attn-split-rows")]
fn row_tiled_attention_op(
    query_rows: u64,
    cached_key_rows: u64,
    kv_heads: u64,
    head_dim: u64,
    cached_lower_inclusive: i64,
) -> (BoundOp, BTreeMap<NodeId, Codec>) {
    let bound = BoundOp {
        node: NodeId(10),
        dtype: DType::FLOAT32,
        extents: vec![query_rows, 1, 8, head_dim],
        kind: BoundOpKind::CachedAttention {
            operands: (0..10)
                .map(|index| {
                    (
                        NodeId(index),
                        Layout {
                            base: 0,
                            strides: vec![1_i64].into(),
                        },
                        None,
                    )
                })
                .collect(),
            query_rows,
            cached_key_rows,
            new_key_rows: query_rows,
            kv_heads,
            query_groups: 8,
            head_dim,
            rotary_dim: head_dim,
            scale: 0.088388346,
            cached_lower_inclusive,
            new_upper_inclusive: 0,
            arithmetic: proxima_tensor::CachedAttentionArithmetic::OnlineF32,
            attention_sink: true, planes: proxima_tensor::bind::ScorePlanes::Paired,
        },
    };
    let packed = [2, 3, 6]
        .into_iter()
        .map(|index| (bound.operands()[index].0, Codec::Float16))
        .collect();
    (bound, packed)
}

#[cfg(feature = "metal-attn-split-rows")]
fn full_layer_op() -> (BoundOp, BTreeMap<NodeId, Codec>) {
    row_tiled_attention_op(512, 2048, 4, 128, i64::MIN)
}

#[cfg(feature = "metal-attn-split-rows")]
fn sliding_layer_op() -> (BoundOp, BTreeMap<NodeId, Codec>) {
    row_tiled_attention_op(512, 2048, 1, 256, -511)
}

#[cfg(feature = "metal-attn-split-rows")]
fn emitted_attention(bound: &BoundOp, packed: &BTreeMap<NodeId, Codec>, coord_index32: bool) -> String {
    let knobs = EmitKnobs::default().with_coord_index32(coord_index32);
    let kernel = emit_with_knobs(bound, packed, NumericPolicy::llama_relaxed(), &knobs)
        .expect("row-tiled attention emits");
    assert!(kernel.entry.contains("_rt"), "not the row-tiled form: {}", kernel.entry);
    kernel.source
}

#[cfg(feature = "metal-attn-split-rows")]
fn attention_cache_key(bound: &BoundOp, packed: &BTreeMap<NodeId, Codec>) -> String {
    kernel_cache_key_with_knobs(
        bound,
        packed,
        NumericPolicy::llama_relaxed(),
        &EmitKnobs::default(),
    )
    .expect("cache key builds")
}

#[cfg(feature = "metal-attn-split-rows")]
#[test]
fn the_captured_full_layer_attention_narrows_to_the_recorded_32_bit_kernel() {
    assert_eq!(narrow_row_tiled_attention(FULL_LAYER_BASE), FULL_LAYER_TIP);
}

#[cfg(feature = "metal-attn-split-rows")]
#[test]
fn the_captured_sliding_layer_attention_narrows_to_the_recorded_32_bit_kernel() {
    assert_eq!(narrow_row_tiled_attention(SLIDING_LAYER_BASE), SLIDING_LAYER_TIP);
}

#[cfg(feature = "metal-attn-split-rows")]
#[test]
fn a_full_layer_attention_renders_the_recorded_fixtures_under_each_coordinate_setting() {
    let (bound, packed) = full_layer_op();

    assert_eq!(emitted_attention(&bound, &packed, false), FULL_LAYER_BASE);
    assert_eq!(emitted_attention(&bound, &packed, true), FULL_LAYER_TIP);
    assert!(attention_cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-attn-split-rows")]
#[test]
fn a_sliding_layer_attention_renders_the_recorded_fixtures_under_each_coordinate_setting() {
    let (bound, packed) = sliding_layer_op();

    assert_eq!(emitted_attention(&bound, &packed, false), SLIDING_LAYER_BASE);
    assert_eq!(emitted_attention(&bound, &packed, true), SLIDING_LAYER_TIP);
    assert!(attention_cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-attn-split-rows")]
#[test]
fn the_narrowed_attention_kernel_drops_every_long_and_the_unbounded_mask_literal() {
    for tip in [FULL_LAYER_TIP, SLIDING_LAYER_TIP] {
        assert!(!has_word(kernel_body(tip), "long"));
        assert!(!tip.contains("-9223372036854775807"));
        assert!(tip.contains("struct Uniforms { long total_elements; long splits; };"));
    }
    assert!(FULL_LAYER_TIP.contains("constexpr int cached_lower = -2147483647;"));
    assert!(SLIDING_LAYER_TIP.contains("constexpr int cached_lower = -511;"));
}

#[cfg(feature = "metal-attn-split-rows")]
#[test]
fn an_attention_cache_past_the_reach_limit_keeps_the_64_bit_kernel() {
    let (bound, packed) = row_tiled_attention_op(512, 1 << 24, 8, 256, i64::MIN);

    let source = emitted_attention(&bound, &packed, true);

    assert_eq!(source, emitted_attention(&bound, &packed, false));
    assert!(has_word(kernel_body(&source), "long"));
    assert!(!attention_cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-attn-split-rows")]
#[test]
fn a_sliding_window_past_the_reach_limit_keeps_the_64_bit_kernel() {
    let (bound, packed) = row_tiled_attention_op(512, 2048, 1, 256, -(1 << 31));

    let source = emitted_attention(&bound, &packed, true);

    assert_eq!(source, emitted_attention(&bound, &packed, false));
    assert!(!attention_cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-attn-split-rows")]
#[test]
fn the_coordinate_override_set_to_zero_restores_the_64_bit_attention_kernel() {
    let (bound, packed) = full_layer_op();

    assert!(has_word(kernel_body(&emitted_attention(&bound, &packed, false)), "long"));
    assert_ne!(
        emitted_attention(&bound, &packed, true),
        emitted_attention(&bound, &packed, false)
    );
}

#[cfg(feature = "metal-attn-variants")]
const FLASH_FULL_LAYER_BASE: &str =
    include_str!("../../tests/fixtures/index32/cached_attention_full_layer_flash.base.msl");
#[cfg(feature = "metal-attn-variants")]
const FLASH_FULL_LAYER_TIP: &str =
    include_str!("../../tests/fixtures/index32/cached_attention_full_layer_flash.tip.msl");
#[cfg(feature = "metal-attn-variants")]
const FLASH_SLIDING_LAYER_BASE: &str =
    include_str!("../../tests/fixtures/index32/cached_attention_sliding_layer_flash.base.msl");
#[cfg(feature = "metal-attn-variants")]
const FLASH_SLIDING_LAYER_TIP: &str =
    include_str!("../../tests/fixtures/index32/cached_attention_sliding_layer_flash.tip.msl");

#[cfg(feature = "metal-attn-variants")]
fn flash_variant() -> AttentionVariant {
    AttentionVariant {
        kv_storage: AttentionKvStorage::F16,
        decomposition: AttentionDecomposition::Flash,
        ..AttentionVariant::default()
    }
}

#[cfg(feature = "metal-attn-variants")]
fn emitted_flash(bound: &BoundOp, packed: &BTreeMap<NodeId, Codec>, coord_index32: bool) -> String {
    let knobs = EmitKnobs::default().with_coord_index32(coord_index32);
    let kernel = emit_with_attention_variant_and_knobs(
        bound,
        packed,
        NumericPolicy::llama_relaxed(),
        flash_variant(),
        &knobs,
    )
    .expect("flash attention emits");
    assert!(kernel.entry.contains("_flash"), "not the flash form: {}", kernel.entry);
    kernel.source
}

/// the key a plan resolves for the flash schedule: the grid and the coordinate bit are both read
/// under the flash row schedule, the way `resolve_step` computes them.
#[cfg(feature = "metal-attn-variants")]
fn flash_cache_key(bound: &BoundOp, packed: &BTreeMap<NodeId, Codec>) -> String {
    let policy = NumericPolicy::llama_relaxed();
    let knobs = EmitKnobs::default();
    let (_, schedule) = AttentionMmaSelection::from_variant(flash_variant()).expect("flash selects");
    let quantized = operand_codecs(bound, packed);
    let grid = grid_spec_with_tile_height(bound, &quantized, policy, false, schedule, &knobs)
        .expect("flash grid");
    kernel_cache_key_for_grid_with_layout(
        bound,
        packed,
        policy,
        &knobs,
        &grid,
        default_dense_mma_input_layout(bound, packed, policy),
        schedule,
    )
    .expect("cache key builds")
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn the_captured_flash_full_layer_attention_narrows_to_the_recorded_32_bit_kernel() {
    assert_eq!(narrow_row_tiled_attention(FLASH_FULL_LAYER_BASE), FLASH_FULL_LAYER_TIP);
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn the_captured_flash_sliding_layer_attention_narrows_to_the_recorded_32_bit_kernel() {
    assert_eq!(narrow_row_tiled_attention(FLASH_SLIDING_LAYER_BASE), FLASH_SLIDING_LAYER_TIP);
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn a_flash_full_layer_renders_the_recorded_fixtures_under_each_coordinate_setting() {
    let (bound, packed) = full_layer_op();

    assert_eq!(emitted_flash(&bound, &packed, false), FLASH_FULL_LAYER_BASE);
    assert_eq!(emitted_flash(&bound, &packed, true), FLASH_FULL_LAYER_TIP);
    assert!(flash_cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn a_flash_sliding_layer_renders_the_recorded_fixtures_under_each_coordinate_setting() {
    let (bound, packed) = sliding_layer_op();

    assert_eq!(emitted_flash(&bound, &packed, false), FLASH_SLIDING_LAYER_BASE);
    assert_eq!(emitted_flash(&bound, &packed, true), FLASH_SLIDING_LAYER_TIP);
    assert!(flash_cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn the_narrowed_flash_kernel_drops_every_long_and_the_unbounded_mask_literal() {
    for tip in [FLASH_FULL_LAYER_TIP, FLASH_SLIDING_LAYER_TIP] {
        assert!(!has_word(kernel_body(tip), "long"));
        assert!(!tip.contains("-9223372036854775807"));
    }
    assert!(FLASH_FULL_LAYER_TIP.contains("constexpr int cached_lower = -2147483647;"));
    assert!(FLASH_SLIDING_LAYER_TIP.contains("constexpr int cached_lower = -511;"));
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn a_flash_cache_past_the_reach_limit_keeps_the_64_bit_kernel() {
    let (bound, packed) = row_tiled_attention_op(512, 1 << 24, 8, 256, i64::MIN);

    let source = emitted_flash(&bound, &packed, true);

    assert_eq!(source, emitted_flash(&bound, &packed, false));
    assert!(has_word(kernel_body(&source), "long"));
    assert!(!flash_cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn a_flash_sliding_window_past_the_reach_limit_keeps_the_64_bit_kernel() {
    let (bound, packed) = row_tiled_attention_op(512, 2048, 1, 256, -(1 << 31));

    let source = emitted_flash(&bound, &packed, true);

    assert_eq!(source, emitted_flash(&bound, &packed, false));
    assert!(has_word(kernel_body(&source), "long"));
    assert!(!flash_cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-grouped-gemm")]
#[test]
fn a_grouped_write_back_stores_four_features_as_one_vector_with_a_scalar_tail_fallback() {
    let vector_store = concat!(
        "*(device float4 *)(out + out_offset) = ",
        "float4(out_vals[0], out_vals[1], out_vals[2], out_vals[3]);"
    );
    let scalar_tail = "out[out_offset + lane_elem * ((int)u.out_strides[2])] = out_vals[lane_elem];";
    for rows in [2048, 2050] {
        let (bound, packed) = flint_down(256, rows, 512);
        let kernel = first_kernel(&emitted(&bound, &packed, true)).to_string();

        assert!(kernel.contains(vector_store), "rows {rows}");
        assert!(kernel.contains("o_valid == 4 && ((int)u.out_strides[2]) == 1"), "rows {rows}");
        assert!(kernel.contains(scalar_tail), "rows {rows}");
        assert!(!has_word(&kernel, "long"), "rows {rows}");
    }
}

#[cfg(feature = "metal-attn-variants")]
const FLASH_FULL_LAYER_F32_QK: &str =
    include_str!("../../tests/fixtures/index32/cached_attention_full_layer_flash.f32.base.msl");
#[cfg(feature = "metal-attn-variants")]
const FLASH_SLIDING_LAYER_F32_QK: &str =
    include_str!("../../tests/fixtures/index32/cached_attention_sliding_layer_flash.f32.base.msl");

#[cfg(feature = "metal-attn-variants")]
fn emitted_flash_with_qk_operand(
    bound: &BoundOp,
    packed: &BTreeMap<NodeId, Codec>,
    operand: Option<FloatLayout>,
) -> String {
    let knobs = EmitKnobs::default().with_coord_index32(false).with_flash_qk_operand(operand);
    emit_with_attention_variant_and_knobs(bound, packed, NumericPolicy::llama_relaxed(), flash_variant(), &knobs)
        .expect("flash attention emits")
        .source
}

/// Pins the binary32-operand flash render against the checked-in fixtures; numerics parity with legacy at f32 is proven by cached_attention_flash_parity, not by this text.
#[cfg(feature = "metal-attn-variants")]
#[test]
fn a_flash_kernel_with_binary32_qk_operands_renders_its_pinned_text() {
    let (full, full_packed) = full_layer_op();
    let (sliding, sliding_packed) = sliding_layer_op();

    assert_eq!(
        emitted_flash_with_qk_operand(&full, &full_packed, Some(FloatLayout::BF32)),
        FLASH_FULL_LAYER_F32_QK
    );
    assert_eq!(
        emitted_flash_with_qk_operand(&sliding, &sliding_packed, Some(FloatLayout::BF32)),
        FLASH_SLIDING_LAYER_F32_QK
    );
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn a_flash_kernel_defaults_to_qk_operands_in_the_k_cache_layout_with_binary32_accumulators() {
    let (bound, packed) = full_layer_op();

    let source = emitted_flash_with_qk_operand(&bound, &packed, None);

    assert!(source.contains("simdgroup_half8x8 qe, qo, kef, kof;"));
    assert!(source.contains("threadgroup half sq["));
    assert!(source.contains("simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>"));
    assert_ne!(source, FLASH_FULL_LAYER_F32_QK);
}
