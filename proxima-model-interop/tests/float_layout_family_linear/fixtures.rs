use std::path::PathBuf;

use omega::msl::layout_render::block_unpack_msl;
use proxima_gguf::quant::block_layout;
use proxima_primitives::Codec;
use proxima_primitives::block_layout::BlockLayout;

use crate::gpu::{Gpu, read_words};

pub const FIXTURE_ELEMENTS: usize = 4096;

pub struct Recorded {
    pub blocks: Vec<u8>,
    pub decoded: Vec<f32>,
}

fn fixture_directory(codec: Codec) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../proxima-tensor/specs/float-layout-dtype/fixtures")
        .join(format!("{codec:?}"))
}

pub fn recorded(codec: Codec) -> Recorded {
    let directory = fixture_directory(codec);
    let blocks = std::fs::read(directory.join("blocks.bin"))
        .unwrap_or_else(|error| panic!("fixture blocks for {codec:?}: {error}"));
    let decoded = std::fs::read(directory.join("decoded_f32.bin"))
        .unwrap_or_else(|error| panic!("fixture decode for {codec:?}: {error}"))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    assert_eq!(
        decoded.len(),
        FIXTURE_ELEMENTS,
        "{codec:?} fixture element count"
    );
    Recorded { blocks, decoded }
}

pub fn layout_of(codec: Codec) -> BlockLayout {
    codec
        .block_layout()
        .unwrap_or_else(|| panic!("{codec:?} is in the linear family"))
}

pub fn mismatches(expected: &[f32], observed: &[f32]) -> usize {
    assert_eq!(
        expected.len(),
        observed.len(),
        "compared sequences differ in length"
    );
    expected
        .iter()
        .zip(observed)
        .filter(|(want, got)| want.to_bits() != got.to_bits())
        .count()
}

pub fn cpu_decode(layout: &BlockLayout, blocks: &[u8]) -> Vec<f32> {
    let mut output = vec![0.0f32; FIXTURE_ELEMENTS];
    block_layout::dequantize(layout, blocks, &mut output).expect("whole blocks decode");
    output
}

pub fn metal_decode(layout: &BlockLayout, blocks: &[u8]) -> Vec<f32> {
    let (elements, block_bytes) = (layout.elements, layout.block_bytes);
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}
kernel void probe(device const uchar *blocks [[buffer(0)]], device float *out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{
    device const uchar *block = blocks + (gid / {elements}u) * {block_bytes}u;
    out[gid] = probe_block(block, gid % {elements}u);
}}
",
        block_unpack_msl(layout, "probe_block")
    );
    let bytes = Gpu::open().run(
        &source,
        "probe",
        &[blocks],
        FIXTURE_ELEMENTS * 4,
        FIXTURE_ELEMENTS,
    );
    read_words(&bytes).into_iter().map(f32::from_bits).collect()
}
