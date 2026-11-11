use alloc::string::String;
#[cfg(all(feature = "metal", target_os = "macos"))]
use core::ffi::c_void;
#[cfg(all(feature = "metal", target_os = "macos"))]
use core::ptr::NonNull;

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_foundation::NSString;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use proxima_numeric::decode_scalar;
use proxima_numeric::FormatCodec;
use proxima_numeric::NumericFormat;
use proxima_numeric::block_layout::BlockLayout;
use proxima_numeric::float_layout::{FloatLayout, NanPolicy, SignEncoding, Specials};

use super::layout_render::{
    block_k_refusal, block_unpack_msl, hardware_type, layout_load_msl, layout_store_msl, readable_block_layout,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use super::layout_render::layout_unpack_msl;

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
    nan: NanPolicy::IEEE,
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
    nan: NanPolicy::IEEE,
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

#[test]
fn ternary_block_layouts_render_from_structural_value_and_scale_fields() {
    let tq1_source = block_unpack_msl(&BlockLayout::TQ1_0, "tq1");
    let tq2_source = block_unpack_msl(&BlockLayout::TQ2_0, "tq2");

    assert!(tq1_source.contains("index >= 0u && index < 160u"));
    assert!(tq1_source.contains("index >= 160u && index < 240u"));
    assert!(tq1_source.contains("index >= 240u && index < 256u"));
    assert!(tq1_source.contains("factor *= 3u"));
    assert!(tq1_source.contains("block[52u]"));
    assert!(tq2_source.contains("block[64u]"));
    assert!(tq2_source.contains("raw_low_rest /= 32u"));
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

const CACHED_ATTENTION_SOURCE: &str = include_str!("cached_attention_render.rs");

fn identifiers_after<'text>(text: &'text str, prefix: &str) -> alloc::vec::Vec<&'text str> {
    text.match_indices(prefix)
        .map(|(match_start, _)| {
            let tail = &text[match_start + prefix.len()..];
            let end = tail
                .find(|current: char| !current.is_alphanumeric() && current != '_')
                .unwrap_or(tail.len());
            &tail[..end]
        })
        .collect()
}

fn function_body<'text>(source: &'text str, signature: &str) -> &'text str {
    let start = source.find(signature).expect("the function is in the source");
    let rest = &source[start..];
    &rest[..rest.find("\n}\n").expect("the function body closes at column 0")]
}

#[test]
fn cached_attention_kv_codec_for_names_no_scalar_format_and_only_the_source_layout() {
    let body = function_body(CACHED_ATTENTION_SOURCE, "pub fn cached_attention_kv_codec_for");
    let format_names = ["Float32", "Float16", "BFloat16", "BFloat8"];

    let named: alloc::vec::Vec<&str> = ["NumericFormat::", "Codec::"]
        .into_iter()
        .flat_map(|prefix| identifiers_after(body, prefix))
        .filter(|name| format_names.contains(name))
        .collect();
    let layouts: alloc::vec::Vec<&str> = identifiers_after(body, "FloatLayout::")
        .into_iter()
        .filter(|name| name.starts_with(|current: char| current.is_ascii_uppercase()))
        .collect();

    assert_eq!(named, alloc::vec::Vec::<&str>::new());
    assert_eq!(layouts, ["BF32"], "the one documented source layout");
}

#[test]
fn cached_attention_kv_codec_for_a_respelled_binary32_is_the_plain_buffer_and_a_narrow_layout_is_not() {
    let respelled_f32 = NumericFormat::Scalar(FloatLayout {
        bias_override: Some(FloatLayout::BF32.bias()),
        ..FloatLayout::BF32
    });

    assert_eq!(super::cached_attention_kv_codec_for(&respelled_f32), Ok(None));
    assert_eq!(super::cached_attention_kv_codec_for(&NumericFormat::Float32), Ok(None));
    assert_eq!(
        super::cached_attention_kv_codec_for(&NumericFormat::BFloat16),
        Ok(Some(NumericFormat::BFloat16))
    );
}

#[test]
fn the_row_tiled_kernel_reads_binary32_bf16_and_bf8_and_refuses_f16_and_quantized_caches() {
    let respelled_f32 = NumericFormat::Scalar(FloatLayout {
        bias_override: Some(127),
        ..FloatLayout::BF32
    });

    assert_eq!(super::cached_attention_row_tiled_kv_codec_for(&NumericFormat::Float32), Ok(None));
    assert_eq!(super::cached_attention_row_tiled_kv_codec_for(&respelled_f32), Ok(None));
    assert_eq!(
        super::cached_attention_row_tiled_kv_codec_for(&NumericFormat::BFloat16),
        Ok(Some(NumericFormat::BFloat16))
    );
    assert_eq!(
        super::cached_attention_row_tiled_kv_codec_for(&NumericFormat::BFloat8),
        Ok(Some(NumericFormat::BFloat8))
    );
    assert!(super::cached_attention_row_tiled_kv_codec_for(&NumericFormat::Float16).is_err());
    assert!(super::cached_attention_row_tiled_kv_codec_for(&NumericFormat::Q8_0).is_err());
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

#[cfg(all(feature = "metal", target_os = "macos"))]
fn random_values(seed: u64, count: usize) -> alloc::vec::Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((state >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let magnitude = [0.02f32, 0.3, 1.0, 7.5][(state >> 20) as usize % 4];
            unit * magnitude
        })
        .collect()
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_block_unpack_kernel(layout: &BlockLayout, blocks: &[u8], elements: usize) -> alloc::vec::Vec<f32> {
    let name = "probe_block";
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}\nkernel void probe(device const uchar *blocks [[buffer(0)]], device float *values [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n    uint block = gid / {}u;\n    values[gid] = {name}(blocks + block * {}u, gid % {}u);\n}}\n",
        block_unpack_msl(layout, name),
        layout.elements,
        layout.block_bytes,
        layout.elements,
    );
    run_probe_source(&source, blocks, elements)
}

/// Runs the `probe` kernel of `source` over `elements` threads: buffer 0 is `input`, buffer 1 the
/// f32 output returned.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_probe_source(source: &str, blocks: &[u8], elements: usize) -> alloc::vec::Vec<f32> {
    let device = MTLCreateSystemDefaultDevice().expect("this box has a Metal device");
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version3_1);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .unwrap_or_else(|error| panic!("metal rejected the source: {error:?}\n{source}"));
    let function = library
        .newFunctionWithName(&NSString::from_str("probe"))
        .expect("the probe kernel is in the compiled library");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|error| panic!("metal could not build a pipeline: {error:?}"));
    // SAFETY: `blocks` outlives the call and the length passed is its own byte length.
    let input = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(blocks.as_ptr().cast_mut().cast::<c_void>()).expect("a slice pointer is never null"),
            blocks.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("the device allocates the block buffer");
    let output = device
        .newBufferWithLength_options(elements * 4, MTLResourceOptions::StorageModeShared)
        .expect("the device allocates the value buffer");
    let queue = device.newCommandQueue().expect("the device makes a command queue");
    let command_buffer = queue.commandBuffer().expect("the queue makes a command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("the command buffer makes an encoder");
    encoder.setComputePipelineState(&pipeline);
    // SAFETY: both buffers are live for the whole encode and index 0 and 1 are the kernel's two buffer slots.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&output), 0, 1);
    }
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: elements, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    // SAFETY: the command buffer completed, so the kernel's writes are visible, and the buffer holds `elements` f32s.
    unsafe { core::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), elements) }.to_vec()
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[proxima::test]
#[case::q4_0_nibble_pairs(NumericFormat::Q4_0)]
#[case::q8_0_whole_bytes(NumericFormat::Q8_0)]
#[case::q5_0_with_a_high_plane(NumericFormat::Q5_0)]
async fn a_rendered_block_unpack_equals_the_host_codec_decode_bit_for_bit_over_4096_blocks(
    #[case] format: NumericFormat,
) {
    let NumericFormat::Block(layout) = format else {
        panic!("{format:?} is a block format");
    };
    let codec = FormatCodec::for_format(&format).expect("the block format resolves a host codec");
    let blocks = 4096usize;
    let elements = blocks * usize::from(layout.elements);
    let values = random_values(0x5eed_0004, elements);
    let mut packed = alloc::vec![0u8; codec.packed_len(elements).expect("whole blocks")];
    codec.encode(&values, &mut packed).expect("host encodes the block format");
    let mut slots = alloc::vec![core::mem::MaybeUninit::<f32>::uninit(); elements];
    codec.decode(&packed, &mut slots).expect("host decodes the block format");
    // decode returned Ok, which initialises every slot of its output
    let expected: alloc::vec::Vec<u32> = slots.iter().map(|slot| unsafe { slot.assume_init() }.to_bits()).collect();

    let observed: alloc::vec::Vec<u32> = run_block_unpack_kernel(&layout, &packed, elements)
        .iter()
        .map(|value| value.to_bits())
        .collect();

    assert_eq!(observed.len(), elements, "every element of 4096 blocks was compared");
    let mismatches = expected.iter().zip(&observed).filter(|(want, got)| want != got).count();
    assert_eq!(mismatches, 0, "rendered MSL decode must equal the host codec decode on every element");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn control_a_corrupted_q4_0_scale_byte_changes_the_rendered_decode() {
    let format = NumericFormat::Q4_0;
    let NumericFormat::Block(layout) = format else {
        panic!("Q4_0 is a block format");
    };
    let codec = FormatCodec::for_format(&format).expect("Q4_0 resolves a host codec");
    let values = random_values(7, 32 * 8);
    let mut packed = alloc::vec![0u8; codec.packed_len(values.len()).expect("whole blocks")];
    codec.encode(&values, &mut packed).expect("host encodes Q4_0");
    let clean = run_block_unpack_kernel(&layout, &packed, values.len());
    packed[1] ^= 0x20;

    let corrupted = run_block_unpack_kernel(&layout, &packed, values.len());

    assert_ne!(clean[..32], corrupted[..32], "the first block's scale feeds its 32 elements");
    assert_eq!(clean[32..], corrupted[32..], "the other blocks are untouched");
}

#[test]
fn a_block_k_beside_a_scalar_v_is_read_and_every_other_pairing_is_refused_naming_the_kernel_capability() {
    let q4_0 = NumericFormat::Q4_0;
    let f16 = NumericFormat::Float16;

    assert_eq!(super::cached_attention_kv_codecs_for(&q4_0, &f16), Ok((Some(q4_0), Some(f16))));
    assert_eq!(super::cached_attention_key_codec_for(&q4_0), Ok(Some(q4_0)));
    assert!(super::cached_attention_kv_codecs_for(&q4_0, &q4_0).is_err(), "a block V has no read path");
    assert!(super::cached_attention_kv_codecs_for(&NumericFormat::Q8_0, &f16).is_ok(), "Q8_0 is a readable block layout too");
    assert!(super::cached_attention_kv_codecs_for(&NumericFormat::Float32, &f16).is_err(), "a plain K beside a narrow V is a mixed pair");
    assert!(super::cached_attention_kv_codecs_for(&q4_0, &NumericFormat::Float32).is_err(), "a block K beside a plain V is a mixed pair");
    assert!(super::cached_attention_kv_codec_for(&q4_0).is_err(), "one block format for both K and V is refused");
    assert!(super::cached_attention_row_tiled_kv_codec_for(&q4_0).is_err(), "row-tiled keeps refusing a block cache");
}

#[test]
fn a_cache_layout_equal_to_the_graph_output_is_the_plain_buffer_whatever_its_spelling() {
    let respelled = NumericFormat::Scalar(FloatLayout {
        bias_override: Some(127),
        ..FloatLayout::BF32
    });

    assert_eq!(super::cached_attention_kv_codec_for(&NumericFormat::Float32), Ok(None));
    assert_eq!(super::cached_attention_kv_codec_for(&respelled), Ok(None));
    assert_eq!(
        super::cached_attention_kv_codec_for(&NumericFormat::Float16),
        Ok(Some(NumericFormat::Float16))
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[proxima::test]
#[case::int16_two_complement(FloatLayout::INT16)]
#[case::int32_two_complement_takes_the_long_arm(FloatLayout::INT32)]
#[case::uint32_unsigned(FloatLayout::UINT32)]
async fn a_rendered_wide_integer_load_equals_the_host_decode_bit_for_bit_on_boundary_codes(
    #[case] layout: FloatLayout,
) {
    let width = u32::from(layout.width());
    let top = if width == 32 { u32::MAX } else { (1u32 << width) - 1 };
    let sign_bit = 1u32 << (width - 1);
    let mut codes: alloc::vec::Vec<u32> = [
        0, 1, 2, top, top - 1, sign_bit, sign_bit - 1, sign_bit + 1, 1 << 15, (1 << 15) - 1, (1 << 15) + 1,
        1 << 30, (1 << 30) - 1, (1 << 30) + 1, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, 0xffff_fffe, 0x1234_5678,
        0xdead_beef,
    ]
    .into_iter()
    .map(|code| code & top)
    .collect();
    codes.sort_unstable();
    codes.dedup();
    let format = NumericFormat::Scalar(layout);
    let expected: alloc::vec::Vec<u32> = codes
        .iter()
        .map(|code| {
            decode_scalar(&format, &code.to_le_bytes()).expect("the host decodes the integer code").to_bits()
        })
        .collect();
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}\nkernel void probe(device const uint *codes [[buffer(0)]], device float *values [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n    values[gid] = probe_load(codes[gid]);\n}}\n",
        layout_load_msl(layout, "probe_load")
    );
    let input: alloc::vec::Vec<u8> = codes.iter().flat_map(|code| code.to_le_bytes()).collect();

    let observed: alloc::vec::Vec<u32> =
        run_probe_source(&source, &input, codes.len()).iter().map(|value| value.to_bits()).collect();

    assert!(codes.len() >= 10, "boundary codes survive the mask and dedup: {codes:?}");
    assert_eq!(observed, expected, "{layout:?} codes {codes:x?}");
    let wide_arm = layout.sign == SignEncoding::TwosComplement && layout.mantissa_bits > 30;
    assert_eq!(source.contains("(long)"), wide_arm, "only a signed layout past 30 mantissa bits takes the long arm");
}

/// the block formats `block_k_refusal` admits: no scale runs, whole four-wide loads, and a host codec
/// that encodes (llama's q4_0, q5_0 and q8_0 k-cache types, and q8_1, which carries a min field).
const ADMITTED_BLOCK_K: [BlockLayout; 4] =
    [BlockLayout::Q4_0, BlockLayout::Q5_0, BlockLayout::Q8_0, BlockLayout::Q8_1];

fn assert_admitted_block_k_set(admitted: &[NumericFormat]) {
    for layout in ADMITTED_BLOCK_K {
        assert!(admitted.contains(&NumericFormat::Block(layout)), "{layout:?} is missing from the admitted set {admitted:?}");
    }
    assert_eq!(admitted.len(), ADMITTED_BLOCK_K.len(), "an unexpected format is admitted: {admitted:?}");
}

fn admitted_block_k_formats() -> alloc::vec::Vec<NumericFormat> {
    NumericFormat::ALL
        .into_iter()
        .filter(|format| readable_block_layout(format).is_some())
        .collect()
}

#[test]
fn every_admitted_block_k_resolves_and_encodes_on_the_host_and_the_admitted_set_is_pinned() {
    let admitted = admitted_block_k_formats();

    for format in &admitted {
        let codec = FormatCodec::for_format(format).expect("an admitted block K resolves a host codec");
        assert!(codec.can_encode(), "{format:?} is admitted so the cache fill can encode it");
    }
    assert_admitted_block_k_set(&admitted);
}

#[test]
fn a_block_format_with_no_host_encode_is_refused_naming_the_missing_encode() {
    let decode_only: alloc::vec::Vec<NumericFormat> = NumericFormat::ALL
        .into_iter()
        .filter(|format| matches!(format, NumericFormat::Block(_)))
        .filter(|format| FormatCodec::for_format(format).is_some_and(|codec| !codec.can_encode()))
        .collect();

    assert!(!decode_only.is_empty(), "the block table holds formats that decode but cannot encode");
    for format in &decode_only {
        let reason = block_k_refusal(format).expect_err("a decode-only block format is not a block K");
        assert!(reason.contains("no encode"), "{format:?} refused with: {reason}");
        assert!(readable_block_layout(format).is_none(), "{format:?} must not be admitted");
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_block_run_kernel(layout: &BlockLayout, blocks: &[u8], elements: usize) -> alloc::vec::Vec<f32> {
    let name = "probe_block";
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}\nkernel void probe(device const uchar *blocks [[buffer(0)]], device float *values [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n    uint block = gid / {}u;\n    uint within = gid % {}u;\n    values[gid] = {name}_run(blocks + block * {}u, within & ~3u)[within & 3u];\n}}\n",
        block_unpack_msl(layout, name),
        layout.elements,
        layout.elements,
        layout.block_bytes,
    );
    run_probe_source(&source, blocks, elements)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn a_rendered_four_element_run_equals_the_host_decode_bit_for_bit_for_every_admitted_block_k_format() {
    let admitted = admitted_block_k_formats();
    assert_admitted_block_k_set(&admitted);

    for format in &admitted {
        let NumericFormat::Block(layout) = format else {
            panic!("{format:?} is a block format");
        };
        let codec = FormatCodec::for_format(format).expect("an admitted block K resolves a host codec");
        let elements = 4096usize * usize::from(layout.elements);
        let values = random_values(0x5eed_0008, elements);
        let mut packed = alloc::vec![0u8; codec.packed_len(elements).expect("whole blocks")];
        codec.encode(&values, &mut packed).expect("host encodes an admitted block K");
        let mut slots = alloc::vec![core::mem::MaybeUninit::<f32>::uninit(); elements];
        codec.decode(&packed, &mut slots).expect("host decodes an admitted block K");
        // decode returned Ok, which initialises every slot of its output
        let expected: alloc::vec::Vec<u32> =
            slots.iter().map(|slot| unsafe { slot.assume_init() }.to_bits()).collect();

        let observed: alloc::vec::Vec<u32> = run_block_run_kernel(layout, &packed, elements)
            .iter()
            .map(|value| value.to_bits())
            .collect();

        assert_eq!(observed.len(), elements, "{format:?}: every element of 4096 blocks was compared");
        let mismatches = expected.iter().zip(&observed).filter(|(want, got)| want != got).count();
        assert_eq!(mismatches, 0, "{format:?}: the rendered run helper must equal the host decode");
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn a_rendered_unpack_equals_the_host_decode_bit_for_bit_for_every_admitted_block_k_format() {
    let admitted = admitted_block_k_formats();
    assert_admitted_block_k_set(&admitted);

    for format in &admitted {
        let NumericFormat::Block(layout) = format else {
            panic!("{format:?} is a block format");
        };
        let codec = FormatCodec::for_format(format).expect("an admitted block K resolves a host codec");
        let blocks = 4096usize;
        let elements = blocks * usize::from(layout.elements);
        let values = random_values(0x5eed_0007, elements);
        let mut packed = alloc::vec![0u8; codec.packed_len(elements).expect("whole blocks")];
        codec.encode(&values, &mut packed).expect("host encodes an admitted block K");
        let mut slots = alloc::vec![core::mem::MaybeUninit::<f32>::uninit(); elements];
        codec.decode(&packed, &mut slots).expect("host decodes an admitted block K");
        // decode returned Ok, which initialises every slot of its output
        let expected: alloc::vec::Vec<u32> =
            slots.iter().map(|slot| unsafe { slot.assume_init() }.to_bits()).collect();

        let observed: alloc::vec::Vec<u32> = run_block_unpack_kernel(layout, &packed, elements)
            .iter()
            .map(|value| value.to_bits())
            .collect();

        assert_eq!(observed.len(), elements, "{format:?}: every element of 4096 blocks was compared");
        let mismatches = expected.iter().zip(&observed).filter(|(want, got)| want != got).count();
        assert_eq!(mismatches, 0, "{format:?}: rendered MSL decode must equal the host decode");
    }
}
