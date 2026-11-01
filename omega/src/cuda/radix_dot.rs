use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_primitives::block_layout::{BlockLayout, Coded};

use super::block_dot::{BlockDot, DOT_ELEMENTS, affine, layout_name, rows_per_warp, scale_helpers};

/// Codes one int8x4 word covers.
const WORD_ELEMENTS: usize = 4;

/// The int8 lowering of a radix-packed [`BlockLayout`]: each code is one mixed-radix digit of a
/// stored byte, recovered as `((byte * radix^lane) mod 256) * radix >> 8`. Four codes that are
/// the same digit of four consecutive bytes form one word, recovered two 16-bit lanes per
/// multiply; any other word is assembled code by code. Derived from the layout's regions and
/// digits alone; no format is named.
pub(super) fn radix_block_dot(layout: &BlockLayout) -> Option<BlockDot> {
    let element = &layout.element;
    let radix = u32::from(element.packed_radix);
    let elements = usize::from(layout.elements);
    let code_bits = u32::BITS - (radix - 1).leading_zeros();
    let usable = layout.sub.is_none()
        && layout.min.is_none()
        && elements.is_multiple_of(DOT_ELEMENTS)
        && radix > 1
        && radix * 255 < 1 << 16
        && codes_fit_int8(element, layout.multiplier, radix);
    if !usable {
        return None;
    }
    let sources = digit_sources(element, radix, elements)?;
    let name = layout_name(layout);
    let mut helpers = scale_helpers(layout, &name)?;
    let words = (0..DOT_ELEMENTS / WORD_ELEMENTS)
        .map(|word| word_text(layout, &sources, radix, code_bits, word))
        .collect::<Option<String>>()?;
    helpers.push_str(&format!(
        "__device__ __forceinline__ void {name}_i8x32(const unsigned char *block, unsigned int chunk, int *words) {{\n{words}}}\n"
    ));
    Some(BlockDot {
        name,
        helpers,
        elements,
        block_bytes: usize::from(layout.block_bytes),
        rows_per_warp: rows_per_warp(true),
    })
}

fn codes_fit_int8(element: &Coded, multiplier: i16, radix: u32) -> bool {
    multiplier > 0
        && (0..radix).all(|code| {
            let value = i32::from(multiplier) * i32::try_from(code).unwrap_or(i32::MAX) - i32::from(element.bias);
            (-128..=127).contains(&value)
        })
}

/// For every element, the byte holding its digit and `radix^lane mod 256`.
fn digit_sources(element: &Coded, radix: u32, elements: usize) -> Option<Vec<(usize, u32)>> {
    let regions = element.regions.get(..usize::from(element.region_count))?;
    (0..elements)
        .map(|index| {
            let region = regions.iter().find(|region| {
                let start = usize::from(region.start);
                (start..start + usize::from(region.elements)).contains(&index)
            })?;
            let local = index - usize::from(region.start);
            let address = region.source.address(local);
            let axis = usize::from(region.digit_axis);
            let divisor: usize = region.source.digits.iter().take(axis).map(|digit| usize::from(digit.radix)).product();
            let lane_radix = usize::from(region.source.digits.get(axis)?.radix);
            let lane = (local / divisor) % lane_radix;
            let factor = (0..lane).fold(1u32, |product, _| (product * radix) % 256);
            (region.source.bits == 8 && address.is_multiple_of(8)).then_some((address / 8, factor))
        })
        .collect()
}

/// `Some((byte, factor))` when the word's four codes are one digit of four consecutive bytes.
fn run_start(sources: &[(usize, u32)]) -> Option<(usize, u32)> {
    let (first_byte, factor) = *sources.first()?;
    sources
        .iter()
        .enumerate()
        .all(|(offset, source)| *source == (first_byte + offset, factor))
        .then_some((first_byte, factor))
}

fn word_text(layout: &BlockLayout, sources: &[(usize, u32)], radix: u32, code_bits: u32, word: usize) -> Option<String> {
    let chunks = usize::from(layout.elements) / DOT_ELEMENTS;
    let per_chunk: Vec<&[(usize, u32)]> = (0..chunks)
        .map(|chunk| {
            let first = chunk * DOT_ELEMENTS + word * WORD_ELEMENTS;
            &sources[first..first + WORD_ELEMENTS]
        })
        .collect();
    let runs: Vec<Option<(usize, u32)>> = per_chunk.iter().map(|codes| run_start(codes)).collect();
    let mut text = String::new();
    let mut branch = "if";
    for (chunk, codes) in per_chunk.iter().enumerate().filter(|(chunk, _)| runs[*chunk].is_none()) {
        let code = mixed_word(codes, radix);
        let value = affine(&code, layout.multiplier, layout.element.bias, code_bits as usize)?;
        text.push_str(&format!("    {branch} (chunk == {chunk}u) {{ words[{word}] = {value}; }}\n"));
        branch = "else if";
    }
    let run_chunks: Vec<(usize, u32)> = runs.iter().flatten().copied().collect();
    if run_chunks.is_empty() {
        return Some(text);
    }
    text.push_str(if branch == "if" { "    {\n" } else { "    else {\n" });
    text.push_str(&run_word(layout, &runs, radix, code_bits, word)?);
    text.push_str("    }\n");
    Some(text)
}

fn mixed_word(codes: &[(usize, u32)], radix: u32) -> String {
    codes
        .iter()
        .enumerate()
        .map(|(position, (byte, factor))| {
            format!(
                "((((((unsigned int)block[{byte}u] * {factor}u) & 0xFFu) * {radix}u) >> 8u) << {}u)",
                position * 8
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn run_word(layout: &BlockLayout, runs: &[Option<(usize, u32)>], radix: u32, code_bits: u32, word: usize) -> Option<String> {
    let present: Vec<(usize, u32)> = runs.iter().flatten().copied().collect();
    let first = *present.first()?;
    let uniform = present.iter().all(|run| *run == first);
    let aligned = usize::from(layout.block_bytes).is_multiple_of(4) && present.iter().all(|(byte, _)| byte.is_multiple_of(4));
    let (byte, factor) = if uniform {
        (format!("{}u", first.0), format!("{}u", first.1))
    } else {
        (format!("(selected{word} & 0xFFu)"), format!("(selected{word} >> 8u)"))
    };
    let mut text = if uniform { String::new() } else { selector(runs, word)? };
    let load = if aligned {
        format!("*(const unsigned int *)(block + {byte})")
    } else {
        (0..4)
            .map(|offset| format!("((unsigned int)block[{byte} + {offset}u] << {}u)", offset * 8))
            .collect::<Vec<_>>()
            .join(" | ")
    };
    let value = affine("(low | (high << 8u))", layout.multiplier, layout.element.bias, code_bits as usize)?;
    text.push_str(&format!(
        "        const unsigned int packed{word} = {load};\n\
         \x20       unsigned int low = ((packed{word} & 0x00FF00FFu) * {factor}) & 0x00FF00FFu;\n\
         \x20       unsigned int high = (((packed{word} >> 8u) & 0x00FF00FFu) * {factor}) & 0x00FF00FFu;\n\
         \x20       low = ((low * {radix}u) >> 8u) & 0x00FF00FFu;\n\
         \x20       high = ((high * {radix}u) >> 8u) & 0x00FF00FFu;\n\
         \x20       words[{word}] = {value};\n"
    ));
    Some(text)
}

/// `selected<word>` = `byte | factor << 8` for the runtime chunk, read out of 16-bit fields
/// packed four to a literal so the table stays in registers.
fn selector(runs: &[Option<(usize, u32)>], word: usize) -> Option<String> {
    let fields: Vec<u64> = runs
        .iter()
        .map(|run| run.map_or(Some(0), |(byte, factor)| u64::try_from(byte).ok().filter(|byte| *byte < 256).map(|byte| byte | u64::from(factor) << 8)))
        .collect::<Option<_>>()?;
    let literals: Vec<u64> = fields
        .chunks(4)
        .map(|group| group.iter().enumerate().fold(0u64, |packed, (slot, field)| packed | field << (16 * slot)))
        .collect();
    let mut table = format!("{:#x}ull", literals[literals.len() - 1]);
    for (index, literal) in literals.iter().enumerate().rev().skip(1) {
        table = format!("(chunk < {}u ? {literal:#x}ull : {table})", (index + 1) * 4);
    }
    Some(format!(
        "        const unsigned int selected{word} = (unsigned int)({table} >> ((chunk & 3u) * 16u)) & 0xFFFFu;\n"
    ))
}
