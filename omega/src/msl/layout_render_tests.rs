use alloc::string::String;

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_foundation::NSString;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{
    MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLLibrary,
};
use proxima_primitives::float_layout::{FloatLayout, SignEncoding, Specials};

use super::layout_render::{hardware_type, layout_load_msl, layout_store_msl, layout_unpack_msl};

const EMITTER_SOURCE: &str = include_str!("layout_render.rs");
const LAYOUT_NAMES: [&str; 12] = [
    "bf16", "bf8", "bf4", "e5m2", "e4m3", "e3m2", "e2m3", "e2m1", "e2m2", "fp6", "f16", "binary32",
];

const E4M3_FROM_FIELDS: FloatLayout = FloatLayout {
    exponent_bits: 4,
    mantissa_bits: 3,
    specials: Specials::NanOnly,
    bias_override: None,
    sign: SignEncoding::Magnitude,
};

fn blank_numeric_literals(source: &str) -> String {
    let mut blanked = String::new();
    let mut previous = ' ';
    let mut chars = source.chars().peekable();
    while let Some(current) = chars.next() {
        let starts_literal =
            current.is_ascii_digit() && !(previous.is_ascii_alphanumeric() || previous == '_');
        if starts_literal {
            while chars
                .next_if(|next| next.is_ascii_alphanumeric() || *next == '.')
                .is_some()
            {}
            blanked.push('#');
            previous = '#';
        } else {
            blanked.push(current);
            previous = current;
        }
    }
    blanked
}

fn rendered(layout: FloatLayout) -> String {
    [
        layout_load_msl(layout, "probe_load"),
        layout_store_msl(layout, "probe_store"),
    ]
    .concat()
}

#[test]
fn float_layout_msl_hardware_type_chosen_by_fields_not_names() {
    let bfloat_by_fields = FloatLayout {
        exponent_bits: 8,
        mantissa_bits: 7,
        specials: Specials::InfinityOnOverflow,
        bias_override: Some(127),
        sign: SignEncoding::Magnitude,
    };
    let bfloat_without_infinity = FloatLayout {
        specials: Specials::None,
        ..bfloat_by_fields
    };

    assert_eq!(hardware_type(&FloatLayout::BF32), Some("float"));
    assert_eq!(hardware_type(&FloatLayout::F16), Some("half"));
    assert_eq!(hardware_type(&bfloat_by_fields), Some("bfloat"));
    assert_eq!(hardware_type(&bfloat_without_infinity), None);
    assert_eq!(hardware_type(&E4M3_FROM_FIELDS), None);
    assert_eq!(hardware_type(&FloatLayout::E3M2), None);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn compile_on_the_metal_compiler(source: &str, entry: &str) {
    let device = MTLCreateSystemDefaultDevice().expect("this box has a Metal device");
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version3_1);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .unwrap_or_else(|error| panic!("metal rejected the source: {error:?}\n{source}"));
    let function = library
        .newFunctionWithName(&NSString::from_str(entry))
        .expect("the probe kernel is in the compiled library");
    device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|error| panic!("metal could not build a pipeline: {error:?}\n{source}"));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn probe_kernel_source(layout: FloatLayout) -> String {
    let flat = layout_unpack_msl(layout, "probe_flat").unwrap_or_default();
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}{flat}
kernel void probe(device const uint *codes [[buffer(0)]], device float *values [[buffer(1)]], device uint *stored [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{
    values[gid] = probe_load(codes[gid]);
    stored[gid] = probe_store(values[gid]);
}}
",
        rendered(layout)
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[proxima::test]
#[case::binary32(FloatLayout::BF32)]
#[case::binary16(FloatLayout::F16)]
#[case::brain16(FloatLayout::BF16)]
#[case::e5m2_saturating(FloatLayout::E5M2_SATURATING)]
#[case::e2m1(FloatLayout::E2M1)]
#[case::e3m2(FloatLayout::E3M2)]
#[case::e2m3(FloatLayout::E2M3)]
#[case::e4m3_from_fields(E4M3_FROM_FIELDS)]
async fn float_layout_msl_source_compiles_on_the_metal_compiler(#[case] layout: FloatLayout) {
    compile_on_the_metal_compiler(&probe_kernel_source(layout), "probe");
}

#[test]
fn float_layout_msl_two_six_bit_layouts_differ_only_in_numeric_constants() {
    let e3m2 = rendered(FloatLayout::E3M2);
    let e2m3 = rendered(FloatLayout::E2M3);

    assert_eq!(FloatLayout::E3M2.width(), FloatLayout::E2M3.width());
    assert_ne!(
        e3m2, e2m3,
        "different fields must render different constants"
    );
    assert_eq!(
        blank_numeric_literals(&e3m2),
        blank_numeric_literals(&e2m3),
        "the two renderings must share one skeleton:\n{e3m2}\n{e2m3}"
    );
}

#[test]
fn float_layout_msl_emitter_source_names_no_layout() {
    let source = EMITTER_SOURCE.to_ascii_lowercase();

    let hits: alloc::vec::Vec<&str> = LAYOUT_NAMES
        .into_iter()
        .filter(|name| source.contains(name))
        .collect();

    assert_eq!(hits, alloc::vec::Vec::<&str>::new());
}

#[test]
fn float_layout_msl_control_changed_bias_changes_the_constants() {
    let shifted = FloatLayout {
        bias_override: Some(4),
        ..FloatLayout::E3M2
    };

    let original = rendered(FloatLayout::E3M2);
    let rebiased = rendered(shifted);

    assert_ne!(original, rebiased);
    assert!(
        original.contains("- (3)"),
        "bias 3 in the decode:\n{original}"
    );
    assert!(
        rebiased.contains("- (4)"),
        "bias 4 in the decode:\n{rebiased}"
    );
    assert_eq!(
        blank_numeric_literals(&original),
        blank_numeric_literals(&rebiased)
    );
}
