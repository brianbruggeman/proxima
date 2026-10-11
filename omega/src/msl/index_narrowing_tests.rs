use alloc::collections::BTreeMap;

use super::tests::embedding_lookup_op;
#[cfg(feature = "metal-grouped-gemm")]
use super::tests::gathered_matmul_op;
use super::*;

const NORM_BASE: &str = include_str!("../../tests/fixtures/index32/norm_epilogue_1536.base.msl");
const NORM_TIP: &str = include_str!("../../tests/fixtures/index32/norm_epilogue_1536.tip.msl");
const COMBINE_BASE: &str = include_str!("../../tests/fixtures/index32/combine_epilogue_1024.base.msl");
const COMBINE_TIP: &str = include_str!("../../tests/fixtures/index32/combine_epilogue_1024.tip.msl");
const EXPERT_BASE: &str = include_str!("../../tests/fixtures/index32/expert_down_q8_0.base.msl");
const EXPERT_TIP: &str = include_str!("../../tests/fixtures/index32/expert_down_q8_0.tip.msl");
const EMBEDDING_BASE: &str = include_str!("../../tests/fixtures/index32/embedding_q6k_1536.base.msl");
const EMBEDDING_TIP: &str = include_str!("../../tests/fixtures/index32/embedding_q6k_1536.tip.msl");
const LOOKUP_BASE: &str = include_str!("../../tests/fixtures/index32/embedding_lookup_f32.base.msl");
const LOOKUP_TIP: &str = include_str!("../../tests/fixtures/index32/embedding_lookup_f32.tip.msl");

fn kernel_body(source: &str) -> &str {
    &source[source.find("kernel void ").expect("a kernel entry")..]
}

fn has_word(text: &str, word: &str) -> bool {
    replace_word(text, word, "") != text
}

fn emitted(bound: &BoundOp, packed: &BTreeMap<NodeId, Codec>, coord_override: Option<&str>) -> String {
    temp_env::with_var("PROXIMA_COORD_INDEX32", coord_override, || {
        emit(bound, packed, NumericPolicy::default()).expect("kernel emits").source
    })
}

fn cache_key(bound: &BoundOp, packed: &BTreeMap<NodeId, Codec>) -> String {
    temp_env::with_var("PROXIMA_COORD_INDEX32", None::<&str>, || {
        kernel_cache_key(bound, packed, NumericPolicy::default()).expect("cache key builds")
    })
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
    let changed = EXPERT_BASE.lines().zip(narrowed.lines()).filter(|(base, tip)| base != tip).count();
    assert_eq!(changed, 2);
}

#[test]
fn the_captured_gathered_embedding_decodes_its_block_index_in_unsigned_32_bit() {
    let narrowed = narrow_block_remainders(&narrow_block_decode(EMBEDDING_BASE));

    assert_eq!(narrowed, EMBEDDING_TIP);
    assert!(narrowed.contains("q6k_element(in0 + (long)((uint)off0 / 256u) * 210, ((uint)off0 % 256u))"));
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

    assert!(narrowed.contains("int span = ((int)u.operand_strides[0][(1)]) * 32 + (int)ulong_value;"), "{narrowed}");
    assert!(narrowed.contains("ulong keep = 7UL;"), "{narrowed}");
    assert!(narrowed.contains("int total = ((int)u.reduction_total);"), "{narrowed}");
}

#[test]
fn a_block_read_that_is_not_an_offset_decode_is_left_alone() {
    let source = "float value = in0[(off0 + 3)] + q4_0_element(in1 + (index / 32) * 18, (uint)(index % 32));";

    assert_eq!(narrow_block_remainders(&narrow_block_decode(source)), source);
}

#[cfg(all(feature = "reduce-epilogue-fusion", feature = "metal-wide-cooperative-reduce"))]
mod norm_rows {
    use super::super::cooperative_fold_tests::normalization_rows::rmsnorm_over_row;
    use super::*;

    const NO_PACKED: BTreeMap<NodeId, Codec> = BTreeMap::new();

    #[test]
    fn a_hidden_width_norm_row_renders_its_index_arithmetic_in_32_bit_under_the_c32_key() {
        let bound = rmsnorm_over_row(1536);

        let source = emitted(&bound, &NO_PACKED, None);

        assert!(!has_word(kernel_body(&source), "long"), "{source}");
        assert!(source.contains("int epi_pre_r = (int)lane + (int)slot * "), "{source}");
        assert!(source.contains("= ((int)u.epilogue_operand_base["), "{source}");
        assert!(source.contains("    long output_total;"));
        assert!(cache_key(&bound, &NO_PACKED).contains("_c32"));
    }

    #[test]
    fn the_coordinate_override_set_to_zero_restores_the_64_bit_norm_body() {
        let bound = rmsnorm_over_row(1536);

        let source = emitted(&bound, &NO_PACKED, Some("0"));

        assert!(source.contains("long epi_pre_r = (long)lane + (long)slot * "), "{source}");
        assert!(!source.contains("int epi_pre_r"));
    }

    #[test]
    fn a_norm_row_exactly_at_the_reach_limit_still_narrows() {
        let bound = rmsnorm_over_row(1 << 30);

        assert!(cache_key(&bound, &NO_PACKED).contains("_c32"));
        assert!(!has_word(kernel_body(&emitted(&bound, &NO_PACKED, None)), "long"));
    }

    #[test]
    fn a_norm_row_one_past_the_reach_limit_keeps_the_64_bit_body_and_drops_the_key_bit() {
        let bound = rmsnorm_over_row((1 << 30) + 1);

        let source = emitted(&bound, &NO_PACKED, None);

        assert!(!cache_key(&bound, &NO_PACKED).contains("_c32"));
        assert!(source.contains("long epi_pre_r = (long)lane + (long)slot * "), "{source}");
        assert!(!source.contains("int epi_pre_r"));
    }
}

#[test]
fn a_gathered_f32_lookup_renders_its_offsets_and_fault_check_in_32_bit() {
    let bound = embedding_lookup_op(50_000, 8, 4);

    let source = emitted(&bound, &BTreeMap::new(), None);

    assert!(!has_word(kernel_body(&source), "long"), "{source}");
    assert!(source.contains("fetched0 < 0 || fetched0 >= ((int)u.gather_extent[0])"), "{source}");
    assert!(cache_key(&bound, &BTreeMap::new()).contains("_c32"));
}

#[test]
fn a_q6_k_embedding_table_past_i32_but_inside_u32_decodes_its_block_in_unsigned_32_bit() {
    let bound = embedding_lookup_op(262_144, 8_960, 1);
    let packed = packed_table(&bound, Codec::Q6K);

    let source = emitted(&bound, &packed, None);

    assert!(source.contains("(long)((uint)off0 / 256u) * 210"), "{source}");
    assert!(source.contains("((uint)off0 % 256u)"), "{source}");
    assert!(!source.contains("(off0 / 256)"));
    assert!(cache_key(&bound, &packed).contains("_c32"));
}

#[test]
fn a_q6_k_embedding_table_past_u32_keeps_the_64_bit_block_decode() {
    let bound = embedding_lookup_op(524_288, 8_960, 1);
    let packed = packed_table(&bound, Codec::Q6K);

    let source = emitted(&bound, &packed, None);

    assert!(source.contains("(off0 / 256) * 210"), "{source}");
    assert!(source.contains("(uint)(off0 % 256)"), "{source}");
    assert!(!cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-grouped-gemm")]
#[test]
fn a_routed_q8_0_expert_projection_divides_its_flat_row_in_unsigned_32_bit() {
    let bound = gathered_matmul_op(8, 32, 512, 1024);
    let packed = packed_table(&bound, Codec::Q8_0);

    let source = emitted(&bound, &packed, None);

    assert!(source.contains("long expert_slot = (long)((uint)group_first / (uint)group_out_extent);"), "{source}");
    assert!(source.contains("long out_row_base = (long)((uint)group_first % (uint)group_out_extent);"), "{source}");
    assert!(cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-grouped-gemm")]
#[test]
fn a_routed_expert_projection_one_row_group_inside_u32_still_narrows() {
    let bound = gathered_matmul_op(63, 2, 67_108_864, 1024);
    let packed = packed_table(&bound, Codec::Q8_0);

    let source = emitted(&bound, &packed, None);

    assert!(source.contains("(uint)group_first / (uint)group_out_extent"), "{source}");
    assert!(cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-grouped-gemm")]
#[test]
fn a_routed_expert_projection_with_outputs_past_u32_keeps_the_64_bit_divides() {
    let bound = gathered_matmul_op(64, 2, 67_108_864, 1024);
    let packed = packed_table(&bound, Codec::Q8_0);

    let source = emitted(&bound, &packed, None);

    assert!(source.contains("long expert_slot = group_first / group_out_extent;"), "{source}");
    assert!(source.contains("long out_row_base = group_first % group_out_extent;"), "{source}");
    assert!(!cache_key(&bound, &packed).contains("_c32"));
}

#[cfg(feature = "metal-grouped-gemm")]
#[test]
fn the_coordinate_override_set_to_zero_restores_every_family_to_the_64_bit_form() {
    let lookup = embedding_lookup_op(50_000, 8, 4);
    let embedding = embedding_lookup_op(262_144, 8_960, 1);
    let expert = gathered_matmul_op(8, 32, 512, 1024);

    let lookup_source = emitted(&lookup, &BTreeMap::new(), Some("0"));
    let embedding_source = emitted(&embedding, &packed_table(&embedding, Codec::Q6K), Some("0"));
    let expert_source = emitted(&expert, &packed_table(&expert, Codec::Q8_0), Some("0"));

    assert!(has_word(kernel_body(&lookup_source), "long"));
    assert!(embedding_source.contains("(off0 / 256) * 210"));
    assert!(expert_source.contains("long expert_slot = group_first / group_out_extent;"));
}
