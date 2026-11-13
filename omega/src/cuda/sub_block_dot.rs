use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_numeric::block_layout::{BlockLayout, Coded, Field, Plane};
use proxima_numeric::float_layout::SignEncoding;
use proxima_numeric::int8_dot::chunk_byte_run;

use super::block_dot::{BlockDot, ChunkTerms, DOT_ELEMENTS, affine, layout_name, rows_per_warp, scale_helpers, scale_read};
use crate::cuda_layout::layout_load_cuda;

const NIBBLE_BITS: u8 = 4;
const BYTE_BITS: usize = 8;
const TABLE_ENTRIES_PER_WORD: usize = 4;
const TABLE_ENTRY_BITS: usize = 16;

/// Where each 32-element chunk of a plane lives: the even byte offset of its first element and the
/// bit shift inside that byte, with consecutive elements one byte apart.
struct ChunkPlane {
    bits: u8,
    bytes: Vec<u32>,
    shifts: Vec<u32>,
}

/// The int8 lowering of a block whose sub-blocks carry their own scale (and optionally min) codes
/// and whose element code is a 4-bit plane plus an optional 1 or 2 bit plane, each chunk's bytes
/// strided one per element at a per-chunk offset and shift. Derived from the planes' address maps
/// alone; no format is named. `parts` is `Int8DotShape::SubBlocks`'s.
pub(super) fn sub_block_dot(layout: &BlockLayout, parts: usize) -> Option<BlockDot> {
    let sub = layout.sub.as_ref()?;
    let elements = usize::from(layout.elements);
    let min = layout.min.as_ref().zip(sub.min.as_ref());
    let chunks = elements / DOT_ELEMENTS;
    let low = chunk_plane(&layout.element.low, chunks)?;
    let high = match &layout.element.high {
        Some(plane) => Some(chunk_plane(plane, chunks)?),
        None => None,
    };
    let code_bits = usize::from(NIBBLE_BITS) + high.as_ref().map_or(0, |plane| usize::from(plane.bits));
    let name = format!("{}_sb{}{}h{}", layout_name(layout), parts, if min.is_some() { "m" } else { "n" }, code_bits - 4);
    let mut helpers = scale_helpers(layout, &name)?;
    helpers.push_str(&words_helper(&name, layout, &low, high.as_ref(), code_bits)?);
    helpers.push_str(&scale_chunk_helpers(&name, sub.scale, usize::from(sub.count), parts)?);
    if let Some((field, coded)) = min {
        helpers.push_str(&min_chunk_helpers(&name, layout, *field, *coded, usize::from(sub.count))?);
    }
    Some(BlockDot {
        name,
        helpers,
        elements,
        block_bytes: usize::from(layout.block_bytes),
        rows_per_warp: rows_per_warp(false),
        centering: None,
        chunk_terms: Some(ChunkTerms { parts, min: min.is_some() }),
    })
}

fn chunk_plane(plane: &Plane, chunks: usize) -> Option<ChunkPlane> {
    let runs = (0..chunks).map(|chunk| chunk_byte_run(plane, chunk)).collect::<Option<Vec<_>>>()?;
    Some(ChunkPlane {
        bits: plane.bits,
        bytes: runs.iter().map(|run| run.0).collect(),
        shifts: runs.iter().map(|run| run.1).collect(),
    })
}

/// A C expression selecting `values[index]`: four 16-bit entries per 64-bit literal.
fn table_expression(values: &[u32], index: &str) -> Option<String> {
    if values.iter().any(|value| *value >= 1 << TABLE_ENTRY_BITS) {
        return None;
    }
    let words: Vec<u64> = values
        .chunks(TABLE_ENTRIES_PER_WORD)
        .map(|group| group.iter().enumerate().fold(0u64, |word, (slot, value)| word | (u64::from(*value) << (TABLE_ENTRY_BITS * slot))))
        .collect();
    let (last, earlier) = words.split_last()?;
    let selected = earlier.iter().enumerate().rev().fold(format!("{last:#x}ull"), |otherwise, (position, word)| {
        format!("(({index}) >> 2u) == {position}u ? {word:#x}ull : {otherwise}")
    });
    Some(format!("(unsigned int)((({selected}) >> ((({index}) & 3u) * {TABLE_ENTRY_BITS}u)) & 0xffffull)"))
}

fn plane_pointer(tag: &str, plane: &ChunkPlane) -> Option<String> {
    let offset = table_expression(&plane.bytes, "chunk")?;
    let shift = table_expression(&plane.shifts, "chunk")?;
    Some(format!(
        "    const unsigned int {tag}_shift = {shift};\n    const unsigned short *{tag}_halves = (const unsigned short *)(block + {offset});\n"
    ))
}

fn plane_word(tag: &str, mask: u32) -> String {
    format!(
        "(((unsigned int){tag}_halves[2 * word] | ((unsigned int){tag}_halves[2 * word + 1] << 16)) >> {tag}_shift) & {mask:#010x}u"
    )
}

fn words_helper(name: &str, layout: &BlockLayout, low: &ChunkPlane, high: Option<&ChunkPlane>, code_bits: usize) -> Option<String> {
    let mut pointers = plane_pointer("low", low)?;
    let mut code = format!("        unsigned int code = {};\n", plane_word("low", 0x0F0F_0F0F));
    if let Some(plane) = high {
        let mask = ((1u32 << plane.bits) - 1) * 0x0101_0101;
        pointers.push_str(&plane_pointer("high", plane)?);
        code.push_str(&format!("        code |= ({}) << 4u;\n", plane_word("high", mask)));
    }
    let word = affine("code", layout.multiplier, layout.element.bias, code_bits)?;
    Some(format!(
        "__device__ __forceinline__ void {name}_i8x32(const unsigned char *block, unsigned int chunk, int *words) {{\n{pointers}    #pragma unroll\n    for (int word = 0; word < 8; word++) {{\n{code}        words[word] = {word};\n    }}\n}}\n"
    ))
}

fn bit_addresses(plane: &Plane, count: usize) -> Option<Vec<u32>> {
    (0..count)
        .map(|sub| {
            let address = plane.address(sub);
            let fits = address % BYTE_BITS + usize::from(plane.bits) <= BYTE_BITS;
            u32::try_from(address).ok().filter(|_| fits)
        })
        .collect()
}

fn field_read(tag: &str, plane: &Plane, count: usize) -> Option<String> {
    let table = table_expression(&bit_addresses(plane, count)?, "sub")?;
    let mask = (1u32 << plane.bits) - 1;
    Some(format!(
        "    const unsigned int {tag}_bit = {table};\n    const unsigned int {tag}_value = ((unsigned int)block[{tag}_bit >> 3u] >> ({tag}_bit & 7u)) & {mask:#x}u;\n"
    ))
}

fn sub_code_helper(function: &str, coded: &Coded, count: usize) -> Option<String> {
    let low_bits = usize::from(coded.low.bits);
    let high_bits = coded.high.as_ref().map_or(0, |plane| usize::from(plane.bits));
    let total = low_bits + high_bits;
    let mut reads = field_read("low", &coded.low, count)?;
    let mut value = String::from("low_value");
    if let Some(plane) = &coded.high {
        reads.push_str(&field_read("high", plane, count)?);
        value = format!("(low_value | (high_value << {low_bits}u))");
    }
    let signed = match coded.code.sign {
        SignEncoding::Absent => format!("(int){value}"),
        SignEncoding::TwosComplement => format!("((int)({value} << {}u)) >> {}", 32 - total, 32 - total),
        SignEncoding::Magnitude => return None,
    };
    Some(format!(
        "__device__ __forceinline__ int {function}(const unsigned char *block, unsigned int sub) {{\n{reads}    return {signed} - ({});\n}}\n",
        coded.bias
    ))
}

fn scale_chunk_helpers(name: &str, coded: Coded, count: usize, parts: usize) -> Option<String> {
    let mut helpers = sub_code_helper(&format!("{name}_sub_scale"), &coded, count)?;
    helpers.push_str(&format!(
        "__device__ __forceinline__ float {name}_chunk_scale(const unsigned char *block, unsigned int chunk, unsigned int part) {{\n    return {name}_scale(block) * (float){name}_sub_scale(block, chunk * {parts}u + part);\n}}\n"
    ));
    Some(helpers)
}

fn min_chunk_helpers(name: &str, layout: &BlockLayout, field: Field, coded: Coded, count: usize) -> Option<String> {
    let read = scale_read(&field)?;
    let mut helpers = sub_code_helper(&format!("{name}_sub_min"), &coded, count)?;
    helpers.push_str(&layout_load_cuda(field.layout, &format!("{name}_min_code")));
    helpers.push_str(&format!(
        "__device__ __forceinline__ float {name}_chunk_min(const unsigned char *block, unsigned int chunk) {{\n    return {}.0f * {name}_min_code({read}) * (float){name}_sub_min(block, chunk);\n}}\n",
        layout.min_multiplier
    ));
    Some(helpers)
}

#[cfg(test)]
mod tests {
    use proxima_numeric::block_layout::BlockLayout;
    use proxima_numeric::int8_dot::{Int8DotShape, int8_dot_shape};

    use crate::cuda::block_dot::{ChunkTerms, block_dot_i8};

    #[test]
    fn k_quants_lower_with_the_chunk_terms_their_sub_blocks_describe() {
        let q4 = block_dot_i8(&BlockLayout::Q4_K).expect("q4_k lowers");
        let q5 = block_dot_i8(&BlockLayout::Q5_K).expect("q5_k lowers");
        let q6 = block_dot_i8(&BlockLayout::Q6_K).expect("q6_k lowers");
        assert_eq!(q4.chunk_terms, Some(ChunkTerms { parts: 1, min: true }));
        assert_eq!(q5.chunk_terms, Some(ChunkTerms { parts: 1, min: true }));
        assert_eq!(q6.chunk_terms, Some(ChunkTerms { parts: 2, min: false }));
        assert_eq!((q4.elements, q4.block_bytes, q5.block_bytes, q6.block_bytes), (256, 144, 176, 210));
        assert!(q5.helpers.contains("high_halves") && !q4.helpers.contains("high_halves"));
    }

    #[test]
    fn layouts_without_a_four_bit_low_plane_or_sub_blocks_are_not_sub_block_codes() {
        for layout in [BlockLayout::Q2_K, BlockLayout::Q3_K, BlockLayout::Q4_0, BlockLayout::Q4_1, BlockLayout::Q8_K] {
            let shape = int8_dot_shape(&layout);
            assert!(!matches!(shape, Some(Int8DotShape::SubBlocks { .. })), "{layout:?} is not a four-bit sub-block code: {shape:?}");
        }
    }
}

#[cfg(all(test, feature = "cuda-driver"))]
mod device_scale_dump {
    use proxima_numeric::block_layout::BlockLayout;
    use proxima_tensor::NodeId;

    use crate::cuda::block_dot::block_dot_i8;
    use crate::cuda::{Binding, CudaGridSpec, CudaKernel};
    use crate::cuda_driver::CudaDriver;

    const QWEN2_BLOB: &str = "/var/lib/ollama/blobs/sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515";
    const QWEN2_BLOB_BYTES: u64 = 397_807_936;
    const PROBES: usize = 16;

    fn first_block(ggml_type: proxima_gguf::GgmlType, block_bytes: usize) -> Vec<u8> {
        let path = proxima_gguf::checkpoint_resolver::require_checkpoint("PROXIMA_QWEN2_MIXED_GGUF", QWEN2_BLOB, QWEN2_BLOB_BYTES);
        let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("the checkpoint reads: {error:?}"));
        let parsed = proxima_gguf::parse_complete(&bytes).unwrap_or_else(|error| panic!("the directory parses: {error:?}"));
        let tensor = parsed.tensors.iter().find(|tensor| tensor.ggml_type == ggml_type).unwrap_or_else(|| panic!("the blob carries the type"));
        let range = parsed.tensor_data_range(tensor, bytes.len() as u64).unwrap_or_else(|error| panic!("the tensor lies in the file: {error:?}"));
        bytes[range.start as usize..][..block_bytes].to_vec()
    }

    fn llama_get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
        }
    }

    fn device_probes(layout: &BlockLayout, block: &[u8], body: &str) -> Vec<f32> {
        let dot = block_dot_i8(layout).unwrap_or_else(|| panic!("the layout lowers"));
        let words: Vec<f32> = block.chunks(4).map(|word| f32::from_bits(u32::from_le_bytes([word[0], word[1], *word.get(2).unwrap_or(&0), *word.get(3).unwrap_or(&0)]))).collect();
        let source = format!(
            "#include <cuda_fp16.h>\n{}\nextern \"C\" __global__ void probe(const float *in, float *out, const int *u) {{\n    const int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i >= {PROBES}) return;\n    const unsigned char *block = (const unsigned char *)in;\n    {body}\n}}\n",
            dot.helpers
        );
        let kernel = CudaKernel {
            source,
            entry: "probe".into(),
            bindings: vec![Binding::Input(NodeId(0)), Binding::Output(NodeId(1)), Binding::Uniforms],
            grid: CudaGridSpec { threads: PROBES as u64, block_width: None },
            extra_outputs: Vec::new(),
            prologue: Vec::new(),
        };
        let name = dot.name;
        let source = kernel.source.replace("NAME", &name);
        let kernel = CudaKernel { source, ..kernel };
        CudaDriver::new(0).unwrap_or_else(|error| panic!("a cuda device: {error:?}")).launch_f32(&kernel, &[&words], &0i64.to_le_bytes()).unwrap_or_else(|error| panic!("the probe launches: {error:?}"))
    }

    #[test]
    fn q4_k_scales_and_mins_decode_to_llama_get_scale_min_k4() {
        let block = first_block(proxima_gguf::GgmlType::Q4_K, 144);
        let (d, dmin) = (half::f16::from_le_bytes([block[0], block[1]]).to_f32(), half::f16::from_le_bytes([block[2], block[3]]).to_f32());
        let scales: [u8; 12] = block[4..16].try_into().unwrap_or_else(|error| panic!("12 scale bytes: {error:?}"));
        let device = device_probes(
            &BlockLayout::Q4_K,
            &block,
            "out[i] = i < 8 ? NAME_chunk_scale(block, i, 0u) : NAME_chunk_min(block, i - 8);",
        );
        for sub in 0..8 {
            let (llama_scale, llama_min) = llama_get_scale_min_k4(sub, &scales);
            let (gguf_scale, gguf_min) = proxima_gguf::quant::q4_k::get_scale_min_k4(sub, &scales);
            println!("q4_k sub {sub}: llama (sc {llama_scale}, m {llama_min}) gguf (sc {gguf_scale}, m {gguf_min}) device scale {} min {} expected scale {} min {}", device[sub], device[8 + sub], d * f32::from(llama_scale), -dmin * f32::from(llama_min));
            assert_eq!((llama_scale, llama_min), (gguf_scale, gguf_min), "sub-block {sub}: gguf walker vs llama");
            assert_eq!(device[sub], d * f32::from(llama_scale), "sub-block {sub}: device scale vs llama");
            assert_eq!(device[8 + sub], -dmin * f32::from(llama_min), "sub-block {sub}: device min vs llama");
        }
    }

    #[test]
    fn q6_k_sub_scales_decode_to_llama_int8_scales_and_the_dequantizer_agrees() {
        let block = first_block(proxima_gguf::GgmlType::Q6_K, 210);
        let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();
        let device = device_probes(&BlockLayout::Q6_K, &block, "out[i] = NAME_chunk_scale(block, i / 2, i % 2);");
        let mut dequantized = [0.0f32; 256];
        proxima_gguf::quant::q6_k::dequantize(&block, &mut dequantized).unwrap_or_else(|error| panic!("one block: {error:?}"));
        for group in 0..16 {
            let llama_scale = f32::from(block[192 + group] as i8);
            println!("q6_k sub {group}: llama int8 scale {llama_scale} device scale {} expected {}", device[group], d * llama_scale);
            assert_eq!(device[group], d * llama_scale, "sub-block {group}: device scale vs llama");
        }
        let (ql, qh) = (&block[..128], &block[128..192]);
        for element in 0..256usize {
            let (half_index, rest) = (element / 128, element % 128);
            let (quarter, lane) = (rest / 32, rest % 32);
            let (low_byte, high_byte) = (ql[half_index * 64 + lane + 32 * (quarter % 2)], qh[half_index * 32 + lane]);
            let low = if quarter < 2 { low_byte & 0xF } else { low_byte >> 4 };
            let value = i32::from(low | (((high_byte >> (2 * quarter)) & 3) << 4)) - 32;
            let llama = d * f32::from(block[192 + element / 16] as i8) * value as f32;
            assert_eq!(dequantized[element], llama, "element {element}: gguf dequantizer vs llama dequantize_row_q6_K");
        }
    }
}
