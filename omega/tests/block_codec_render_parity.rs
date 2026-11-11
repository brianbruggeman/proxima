//! The rendered `q4_0_element`, `q5_0_element` and `q5_1_element` accessors against the output of
//! the hand-written MSL strings they replaced (PROXNUM-58), as raw f32 bit patterns on Metal.
//!
//! The hand output was captured once, before the strings were deleted, into
//! `tests/fixtures/block_codec_hand/<Format>/` (`blocks.bin` = 128 real blocks read from a GGUF on
//! this host, `hand_f32.bin` = what the hand string wrote for those 4096 elements, `manifest.toml`
//! = the blob, tensor and byte offset). The fixtures are checked in, so this test does not depend
//! on the host checkpoint being present.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "../../proxima-model-interop/tests/float_layout_parity/gpu.rs"]
mod gpu;

use std::path::PathBuf;

use omega::msl::layout_render::block_element_msl;
use proxima_numeric::block_layout::BlockLayout;

use gpu::{Gpu, read_words};

const ELEMENTS: usize = 4096;
const BLOCK_ELEMENTS: usize = 32;

struct Fixture {
    blocks: Vec<u8>,
    hand: Vec<u32>,
}

fn fixture(format: &str) -> Fixture {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/block_codec_hand")
        .join(format);
    let blocks = std::fs::read(directory.join("blocks.bin")).expect("fixture blocks are checked in");
    let hand = read_words(&std::fs::read(directory.join("hand_f32.bin")).expect("fixture hand output is checked in"));
    Fixture { blocks, hand }
}

fn metal_bits(source: &str, name: &str, block_bytes: usize, blocks: &[u8]) -> Vec<u32> {
    let program = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{source}\nkernel void probe(device const uchar *blocks [[buffer(0)]], device float *out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n    device const uchar *block = blocks + (gid / {BLOCK_ELEMENTS}u) * {block_bytes}u;\n    out[gid] = {name}(block, gid % {BLOCK_ELEMENTS}u);\n}}\n"
    );
    read_words(&Gpu::open().run(&program, "probe", &[blocks], ELEMENTS * 4, ELEMENTS))
}

fn mismatches(hand: &[u32], rendered: &[u32]) -> usize {
    assert_eq!(hand.len(), ELEMENTS, "degenerate gate: hand fixture length");
    assert_eq!(rendered.len(), ELEMENTS, "degenerate gate: rendered length");
    hand.iter().zip(rendered).filter(|(want, got)| want != got).count()
}

fn assert_rendered_equals_hand(format: &str, rendered_source: &str, name: &str, block_bytes: usize) {
    let recorded = fixture(format);
    assert_eq!(recorded.blocks.len(), ELEMENTS / BLOCK_ELEMENTS * block_bytes);
    let distinct = {
        let mut sorted = recorded.hand.clone();
        sorted.sort_unstable();
        sorted.dedup();
        sorted.len()
    };
    assert!(distinct > 16, "{format}: hand fixture holds {distinct} distinct values, too few to discriminate");

    let rendered = metal_bits(rendered_source, name, block_bytes, &recorded.blocks);

    assert_eq!(mismatches(&recorded.hand, &rendered), 0, "{format}: rendered bits differ from the hand string's");
}

fn assert_changed_layout_fails_parity(format: &str, changed: &BlockLayout, name: &str, block_bytes: usize) {
    let recorded = fixture(format);

    let rendered = metal_bits(&block_element_msl(changed, name), name, block_bytes, &recorded.blocks);

    assert!(mismatches(&recorded.hand, &rendered) > 0, "{format}: a changed layout field still matched the hand output");
}

#[test]
fn q4_0_rendered_unpack_equals_hand_string_bits_on_real_blocks() {
    assert_rendered_equals_hand("Q4_0", &omega::q4_0_unpack_msl(), "q4_0_element", 18);
}

#[test]
fn q5_0_rendered_unpack_equals_hand_string_bits_on_real_blocks() {
    assert_rendered_equals_hand("Q5_0", &omega::q5_0_unpack_msl(), "q5_0_element", 22);
}

#[test]
fn q5_1_rendered_unpack_equals_hand_string_bits_on_real_blocks() {
    assert_rendered_equals_hand("Q5_1", &omega::q5_1_unpack_msl(), "q5_1_element", 24);
}

#[test]
fn control_q4_0_with_bias_seven_fails_parity() {
    let mut changed = BlockLayout::Q4_0;
    changed.element.bias = 7;
    assert_changed_layout_fails_parity("Q4_0", &changed, "q4_0_element", 18);
}

#[test]
fn control_q5_0_with_bias_fifteen_fails_parity() {
    let mut changed = BlockLayout::Q5_0;
    changed.element.bias = 15;
    assert_changed_layout_fails_parity("Q5_0", &changed, "q5_0_element", 22);
}

#[test]
fn control_q5_1_with_subtracted_min_fails_parity() {
    let mut changed = BlockLayout::Q5_1;
    changed.min_multiplier = -1;
    assert_changed_layout_fails_parity("Q5_1", &changed, "q5_1_element", 24);
}

#[test]
fn hand_written_unpack_strings_number_seven() {
    let source = include_str!("../src/msl/kernel_types_identity.rs");
    let count = source
        .lines()
        .filter(|line| line.starts_with("pub const ") && line.contains("_UNPACK_MSL: &str"))
        .count();
    assert_eq!(count, 7, "hand _UNPACK_MSL strings: 10 before PROXNUM-58, 7 after");
}
