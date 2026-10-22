use alloc::collections::BTreeMap;

use proxima_primitives::float_layout::{FloatLayout, SignEncoding, Specials};
use proxima_tensor::{
    BoundOp, DType, Extent, IndexMap, NodeId, NumericPolicy, Op, ScalarOp, append, bind, infer, map,
};

use crate::cuda::emit_cuda;
use crate::cuda_layout::{layout_load_cuda, layout_store_cuda};
use crate::wgsl::{WgslCaps, emit_wgsl};
use crate::wgsl_layout::{layout_load_wgsl, layout_store_wgsl};

const E4M3_FROM_FIELDS: FloatLayout = FloatLayout {
    exponent_bits: 4,
    mantissa_bits: 3,
    specials: Specials::NanOnly,
    bias_override: None,
    sign: SignEncoding::Magnitude,
};

const BINARY16_BY_FIELDS: FloatLayout = FloatLayout {
    exponent_bits: 5,
    mantissa_bits: 10,
    specials: Specials::InfinityOnOverflow,
    bias_override: Some(15),
    sign: SignEncoding::Magnitude,
};

const BRAIN16_BY_FIELDS: FloatLayout = FloatLayout {
    exponent_bits: 8,
    mantissa_bits: 7,
    specials: Specials::InfinityOnOverflow,
    bias_override: Some(127),
    sign: SignEncoding::Magnitude,
};

fn negate_bound(layout: FloatLayout) -> BoundOp {
    let dtype = DType::Layout(layout);
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype,
            shape: vec![Extent::Static(4), Extent::Static(8)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype,
            body: ScalarOp::Negate,
            operands: vec![(source, IndexMap::Affine(map::projection(2, &[0, 1])))],
            name: None,
        },
    );
    let shapes = infer(&program, &[]).expect("the negate program infers");
    let terminal = NodeId((program.len() - 1) as u32);
    bind(&program, &shapes, &[terminal], NumericPolicy::default())
        .expect("the negate program lowers")
        .into_iter()
        .next()
        .expect("one bound op")
}

fn cuda_source(layout: FloatLayout) -> String {
    emit_cuda(&negate_bound(layout), &BTreeMap::new())
        .expect("a float layout emits CUDA without a typed rejection")
        .source
}

fn wgsl_source(layout: FloatLayout) -> String {
    let caps = WgslCaps {
        shader_f16: true,
        ..WgslCaps::default()
    };
    emit_wgsl(&negate_bound(layout), caps, &BTreeMap::new())
        .expect("a float layout emits WGSL without a typed rejection")
        .source
}

#[test]
fn e4m3fn_lowerers_reserve_only_the_top_nan_code() {
    let msl_load = crate::msl::layout_render::layout_load_msl(FloatLayout::E4M3FN, "load");
    let msl_store = crate::msl::layout_render::layout_store_msl(FloatLayout::E4M3FN, "store");
    let wgsl_load = layout_load_wgsl(FloatLayout::E4M3FN, "load");
    let wgsl_store = layout_store_wgsl(FloatLayout::E4M3FN, "store");

    assert!(
        msl_load.contains("exponent == 0xfu && mantissa == 0x7u"),
        "{msl_load}"
    );
    assert!(
        wgsl_load.contains("exponent == 0xfu && mantissa == 0x7u"),
        "{wgsl_load}"
    );
    assert!(msl_load.contains("mantissa << 20u"), "{msl_load}");
    assert!(wgsl_load.contains("mantissa << 20u"), "{wgsl_load}");
    assert!(
        msl_store.contains("return sign | 0x78u | 0x7u"),
        "{msl_store}"
    );
    assert!(
        wgsl_store.contains("return sign | 0x78u | 0x7u"),
        "{wgsl_store}"
    );
}

#[test]
fn cuda_and_wgsl_refuse_layouts_wider_than_the_generic_codec() {
    let wide = negate_bound(FloatLayout::F64);

    assert!(emit_cuda(&wide, &BTreeMap::new()).is_err());
    assert!(emit_wgsl(&wide, WgslCaps::default(), &BTreeMap::new()).is_err());
}

fn assert_wgsl_validates(load: &str, store: &str) {
    let source = format!(
        "{load}{store}
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn probe_entry(@builtin(global_invocation_id) id: vec3<u32>) {{
    data[id.x] = probe_store(probe_load(data[id.x]));
}}
"
    );
    let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
        panic!(
            "naga rejects the wgsl: {}\n{source}",
            error.emit_to_string(&source)
        )
    });
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("naga's validator rejects the wgsl: {error:?}\n{source}"));
}

fn cuda_output_declaration(token: &str) -> String {
    format!("    {token}* __restrict__ out,")
}

#[proxima::test]
#[case::binary32(FloatLayout::BF32, "float")]
#[case::binary16(FloatLayout::F16, "__half")]
#[case::brain16(FloatLayout::BF16, "__nv_bfloat16")]
#[case::e5m2_saturating(FloatLayout::E5M2_SATURATING, "float")]
#[case::e2m1(FloatLayout::E2M1, "float")]
#[case::e3m2(FloatLayout::E3M2, "float")]
#[case::e2m3(FloatLayout::E2M3, "float")]
#[case::e4m3_from_fields(E4M3_FROM_FIELDS, "float")]
async fn float_layout_cuda_emits_text_for_layout(#[case] layout: FloatLayout, #[case] token: &str) {
    let source = cuda_source(layout);
    let load = layout_load_cuda(layout, "probe_load");
    let store = layout_store_cuda(layout, "probe_store");

    assert!(source.contains(&cuda_output_declaration(token)), "{source}");
    assert!(load.contains("__device__ __forceinline__ float probe_load(unsigned int code)"));
    assert!(store.contains("__device__ __forceinline__ unsigned int probe_store(float value)"));
    assert!(
        !load.contains("uint ") && !store.contains("uint "),
        "no MSL type left:\n{load}\n{store}"
    );
}

#[proxima::test]
#[case::binary32(FloatLayout::BF32, false)]
#[case::binary16(FloatLayout::F16, true)]
#[case::brain16(FloatLayout::BF16, false)]
#[case::e5m2_saturating(FloatLayout::E5M2_SATURATING, false)]
#[case::e2m1(FloatLayout::E2M1, false)]
#[case::e3m2(FloatLayout::E3M2, false)]
#[case::e2m3(FloatLayout::E2M3, false)]
#[case::e4m3_from_fields(E4M3_FROM_FIELDS, false)]
async fn float_layout_wgsl_emits_text_for_layout(
    #[case] layout: FloatLayout,
    #[case] native: bool,
) {
    let source = wgsl_source(layout);
    let load = layout_load_wgsl(layout, "probe_load");
    let store = layout_store_wgsl(layout, "probe_store");

    assert_eq!(source.starts_with("enable f16;\n"), native, "{source}");
    assert!(load.contains("fn probe_load(code: u32) -> f32"));
    assert!(store.contains("fn probe_store(value: f32) -> u32"));
    assert_wgsl_validates(&load, &store);
}

#[test]
fn float_layout_cuda_native_types_chosen_by_fields() {
    let brain16_without_infinity = FloatLayout {
        specials: Specials::None,
        ..BRAIN16_BY_FIELDS
    };

    let half = cuda_source(BINARY16_BY_FIELDS);
    let brain = cuda_source(BRAIN16_BY_FIELDS);
    let near_miss = cuda_source(brain16_without_infinity);
    let unnamed = cuda_source(E4M3_FROM_FIELDS);

    assert!(half.contains(&cuda_output_declaration("__half")));
    assert!(brain.contains(&cuda_output_declaration("__nv_bfloat16")));
    assert!(brain.contains("#include <cuda_bf16.h>"));
    assert!(near_miss.contains(&cuda_output_declaration("float")));
    assert!(unnamed.contains(&cuda_output_declaration("float")));
    assert!(!unnamed.contains("#include <cuda_bf16.h>"));
    assert!(layout_load_cuda(BINARY16_BY_FIELDS, "probe_load").contains("__half2float"));
    assert!(layout_load_cuda(BRAIN16_BY_FIELDS, "probe_load").contains("__bfloat162float"));
    assert!(!layout_load_cuda(E4M3_FROM_FIELDS, "probe_load").contains("__half"));
}

#[test]
fn float_layout_wgsl_native_types_chosen_by_fields() {
    let without_f16 = emit_wgsl(
        &negate_bound(BINARY16_BY_FIELDS),
        WgslCaps::default(),
        &BTreeMap::new(),
    );

    assert!(wgsl_source(BINARY16_BY_FIELDS).starts_with("enable f16;\n"));
    assert!(!wgsl_source(BRAIN16_BY_FIELDS).contains("enable f16"));
    assert!(!wgsl_source(E4M3_FROM_FIELDS).contains("enable f16"));
    assert!(
        without_f16.is_err(),
        "binary16 fields need the adapter's shader-f16"
    );
}

#[test]
fn float_layout_cuda_wgsl_control_changed_bias_changes_constants() {
    let shifted = FloatLayout {
        bias_override: Some(4),
        ..FloatLayout::E3M2
    };

    let cuda = layout_load_cuda(FloatLayout::E3M2, "probe_load");
    let cuda_shifted = layout_load_cuda(shifted, "probe_load");
    let wgsl = layout_load_wgsl(FloatLayout::E3M2, "probe_load");
    let wgsl_shifted = layout_load_wgsl(shifted, "probe_load");

    assert_ne!(cuda, cuda_shifted);
    assert_ne!(wgsl, wgsl_shifted);
    assert!(cuda.contains("- (3)") && cuda_shifted.contains("- (4)"));
    assert!(wgsl.contains("- (3)") && wgsl_shifted.contains("- (4)"));
}

#[test]
fn integer_wgsl_text_validates_in_naga() {
    for layout in [
        FloatLayout::INT8,
        FloatLayout::UINT8,
        FloatLayout::INT16,
        FloatLayout::UINT16,
        FloatLayout::SIGN_MAGNITUDE_INT8,
    ] {
        assert_wgsl_validates(
            &layout_load_wgsl(layout, "probe_load"),
            &layout_store_wgsl(layout, "probe_store"),
        );
    }
}
