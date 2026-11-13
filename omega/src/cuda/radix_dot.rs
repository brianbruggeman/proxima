use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_numeric::block_layout::{BlockLayout, Coded};
use proxima_numeric::int8_dot::radix_digit_source;

use super::block_dot::{BlockDot, DOT_ELEMENTS, layout_name, rows_per_warp, scale_helpers};

/// Codes one int8x4 word covers.
const WORD_ELEMENTS: usize = 4;

/// The int8 lowering of a radix-packed [`BlockLayout`]: each code is one mixed-radix digit of a
/// stored byte, recovered as `((byte * radix^lane) mod 256) * radix >> 8`. Four codes that are
/// the same digit of four consecutive bytes form one word, recovered two 16-bit lanes per
/// multiply; any other word is assembled code by code. Derived from the layout's regions and
/// digits alone; no format is named. `radix` is `Int8DotShape::Radix`'s.
pub(super) fn radix_block_dot(layout: &BlockLayout, radix: u32) -> Option<BlockDot> {
    let element = &layout.element;
    let elements = usize::from(layout.elements);
    let sources = digit_sources(element, radix, elements)?;
    let name = layout_name(layout);
    let mut helpers = scale_helpers(layout, &name)?;
    let words = (0..DOT_ELEMENTS / WORD_ELEMENTS)
        .map(|word| word_text(layout, &sources, radix, word))
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
        centering: Some((layout.multiplier, element.bias)),
        chunk_terms: None,
    })
}

/// For every element, the byte holding its digit and `radix^lane mod 256`.
fn digit_sources(element: &Coded, radix: u32, elements: usize) -> Option<Vec<(usize, u32)>> {
    (0..elements).map(|index| radix_digit_source(element, radix, index)).collect()
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

fn word_text(layout: &BlockLayout, sources: &[(usize, u32)], radix: u32, word: usize) -> Option<String> {
    let chunks = usize::from(layout.elements) / DOT_ELEMENTS;
    let per_chunk: Vec<&[(usize, u32)]> = (0..chunks)
        .map(|chunk| {
            let first = chunk * DOT_ELEMENTS + word * WORD_ELEMENTS;
            &sources[first..first + WORD_ELEMENTS]
        })
        .collect();
    let runs: Vec<Option<(usize, u32)>> = per_chunk.iter().map(|codes| run_start(codes)).collect();
    if let Some(text) = uniform_word(layout, &per_chunk, &runs, radix, word) {
        return Some(text);
    }
    let mut text = String::new();
    let mut branch = "if";
    for (chunk, codes) in per_chunk.iter().enumerate().filter(|(chunk, _)| runs[*chunk].is_none()) {
        let code = mixed_word(codes, radix);
        text.push_str(&format!("    {branch} (chunk == {chunk}u) {{ words[{word}] = {code}; }}\n"));
        branch = "else if";
    }
    let run_chunks: Vec<(usize, u32)> = runs.iter().flatten().copied().collect();
    if run_chunks.is_empty() {
        return Some(text);
    }
    text.push_str(if branch == "if" { "    {\n" } else { "    else {\n" });
    text.push_str(&run_word(layout, &runs, radix, word)?);
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

fn run_word(layout: &BlockLayout, runs: &[Option<(usize, u32)>], radix: u32, word: usize) -> Option<String> {
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
    text.push_str(&format!(
        "        const unsigned int packed{word} = {load};\n\
         \x20       unsigned int low = ((packed{word} & 0x00FF00FFu) * {factor}) & 0x00FF00FFu;\n\
         \x20       unsigned int high = (((packed{word} >> 8u) & 0x00FF00FFu) * {factor}) & 0x00FF00FFu;\n\
         \x20       words[{word}] = (int)__byte_perm(low * {radix}u, high * {radix}u, 0x7351u);\n"
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

/// One chunk position's way of decoding a word: which aligned 32-bit load, which bytes of it feed
/// the even and odd 16-bit lanes, and the per-lane multipliers.
#[derive(Clone, Copy)]
struct WordSpec {
    base: usize,
    even_shift: u32,
    odd_shift: u32,
    mask: u32,
    even_factor: u32,
    odd_factor: u32,
}

fn run_spec((byte, factor): (usize, u32)) -> WordSpec {
    WordSpec { base: byte, even_shift: 0, odd_shift: 8, mask: 0x00FF_00FF, even_factor: factor, odd_factor: factor }
}

/// Codes 0 and 2 share one byte and codes 1 and 3 share another, both inside one aligned word:
/// each byte is multiplied by both of its lanes' factors in a single multiply.
fn pair_spec(codes: &[(usize, u32)]) -> Option<WordSpec> {
    let [(even_byte, first), (odd_byte, second), (even_again, third), (odd_again, fourth)] = *codes else {
        return None;
    };
    let same_word = even_byte / 4 == odd_byte / 4;
    (even_byte == even_again && odd_byte == odd_again && same_word && first < 256 && second < 256 && third < 256 && fourth < 256).then_some(WordSpec {
        base: even_byte / 4 * 4,
        even_shift: (even_byte % 4) as u32 * 8,
        odd_shift: (odd_byte % 4) as u32 * 8,
        mask: 0xFF,
        even_factor: first | third << 16,
        odd_factor: second | fourth << 16,
    })
}

/// Every chunk position decodes with the same instruction stream, its parameters picked out of
/// literals by the lane-invariant position, when each is a run or a byte pair.
fn uniform_word(
    layout: &BlockLayout,
    per_chunk: &[&[(usize, u32)]],
    runs: &[Option<(usize, u32)>],
    radix: u32,
    word: usize,
) -> Option<String> {
    let aligned = usize::from(layout.block_bytes).is_multiple_of(4);
    let specs: Vec<WordSpec> = per_chunk
        .iter()
        .zip(runs)
        .map(|(codes, run)| run.map(run_spec).or_else(|| pair_spec(codes)))
        .collect::<Option<_>>()?;
    let any_pair = runs.iter().any(Option::is_none);
    let loadable = specs.iter().all(|spec| spec.base.is_multiple_of(4));
    if !(aligned && any_pair && loadable) {
        return None;
    }
    let pick = |value: fn(&WordSpec) -> u32| select_by_position(&specs.iter().map(value).collect::<Vec<_>>());
    Some(format!(
        "    {{\n\
         \x20       const unsigned int packed{word} = *(const unsigned int *)(block + {base});\n\
         \x20       unsigned int low = (((packed{word} >> {even_shift}) & {mask}) * {even_factor}) & 0x00FF00FFu;\n\
         \x20       unsigned int high = (((packed{word} >> {odd_shift}) & {mask}) * {odd_factor}) & 0x00FF00FFu;\n\
         \x20       words[{word}] = (int)__byte_perm(low * {radix}u, high * {radix}u, 0x7351u);\n\
         \x20   }}\n",
        base = pick(|spec| spec.base as u32),
        even_shift = pick(|spec| spec.even_shift),
        odd_shift = pick(|spec| spec.odd_shift),
        mask = pick(|spec| spec.mask),
        even_factor = pick(|spec| spec.even_factor),
        odd_factor = pick(|spec| spec.odd_factor),
    ))
}

fn select_by_position(values: &[u32]) -> String {
    let last = values[values.len() - 1];
    values
        .iter()
        .enumerate()
        .rev()
        .skip(1)
        .fold(format!("{last}u"), |otherwise, (position, value)| {
            if format!("{value}u") == otherwise {
                otherwise
            } else {
                format!("(chunk == {position}u ? {value}u : {otherwise})")
            }
        })
}
