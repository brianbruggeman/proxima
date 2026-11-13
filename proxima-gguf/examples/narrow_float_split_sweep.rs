//! PROXNUM-109: the exponent/mantissa split of narrow float formats on real GGUF tensors, accuracy
//! half, and the describe/encode probe of each split.
//!
//! Every split is one description: an element [`FloatLayout`] (exponent bits, mantissa bits, specials
//! rule), and optionally a block scale [`FloatLayout`] with a block size. Nothing here names a
//! format; labels are spelled from the fields. The element and scale rounding is
//! [`FormatCodec::scalar`] (host encode) and [`decode_blocks`] (decode), so a split's error is the
//! error of the codec the workspace already ships, not of a second quantizer written here.
//!
//! Usage:
//! `cargo run --release -p proxima-gguf --example narrow_float_split_sweep -- <gguf> accuracy`
//! `cargo run --release -p proxima-gguf --example narrow_float_split_sweep -- <gguf> support`
//!
//! `narrow_float_split_sweep raw <file.f32> <role> [<tensor-label>]` runs the same sweep over a
//! little-endian f32 file (captured K/V or activations); `gen-normal <out.f32> [<count>]` writes a
//! deterministic N(0,1) file to check it against.
//!
//! `accuracy` prints one TSV row per (tensor, split); `support` prints one row per split saying whether
//! a [`BlockLayout`] describes it, whether the layout decodes real bytes the way the host quantizer
//! meant, and which host encode routes accept it.

use std::env;
use std::error::Error;
use std::fs::{self, File};

use memmap2::Mmap;
use proxima_gguf::pipe::{ParsedGguf, parse_complete};
use proxima_gguf::quant::QuantError;
use proxima_gguf::quant::dispatch::dequantize;
use proxima_gguf::quant::{bf16, f16, q4_0, q4_k, q5_k, q6_k, q8_0};
use proxima_gguf::types::GgmlType;
use proxima_numeric::block_layout::{Coded, Digit, Field, Plane};
use proxima_numeric::float_layout::{NanPolicy, SignEncoding, Specials};
use proxima_numeric::{BlockLayout, FloatLayout, FormatCodec, NumericFormat, decode_blocks, encode_blocks};

const PEAK_GROWTH_PASSES: usize = 8;

type Quantizer = fn(&[f32], &mut [u8]) -> Result<(), QuantError>;

/// One weight window to measure: `rows` rows of the tensor's first dimension, from `first_row`.
struct Window {
    role: &'static str,
    tensor: &'static str,
    first_row: usize,
    rows: usize,
}

const WINDOWS: [Window; 12] = [
    Window { role: "attn_q", tensor: "blk.3.attn_q.weight", first_row: 0, rows: 1024 },
    Window { role: "attn_output", tensor: "blk.3.attn_output.weight", first_row: 0, rows: 512 },
    Window { role: "expert_gate", tensor: "blk.2.ffn_gate_exps.weight", first_row: 0, rows: 512 },
    Window { role: "expert_gate", tensor: "blk.3.ffn_gate_exps.weight", first_row: 0, rows: 512 },
    Window { role: "expert_gate", tensor: "blk.20.ffn_gate_exps.weight", first_row: 0, rows: 512 },
    Window { role: "expert_gate", tensor: "blk.21.ffn_gate_exps.weight", first_row: 0, rows: 512 },
    Window { role: "expert_down", tensor: "blk.4.ffn_down_exps.weight", first_row: 0, rows: 2048 },
    Window { role: "expert_down", tensor: "blk.8.ffn_down_exps.weight", first_row: 0, rows: 2048 },
    Window { role: "expert_down", tensor: "blk.19.ffn_down_exps.weight", first_row: 0, rows: 2048 },
    Window { role: "expert_down", tensor: "blk.3.ffn_down_exps.weight", first_row: 0, rows: 2048 },
    Window { role: "shared_expert", tensor: "blk.3.ffn_gate_shexp.weight", first_row: 0, rows: 1024 },
    Window { role: "token_embd", tensor: "token_embd.weight", first_row: 1000, rows: 512 },
];

/// The element widths of the sweep as (exponent bits, mantissa bits).
const ELEMENT_SPLITS: [(u8, u8); 8] = [(2, 1), (3, 0), (1, 2), (3, 2), (2, 3), (4, 1), (4, 3), (5, 2)];

/// A block scale: its own layout, how many elements share it, whether a per-tensor factor first
/// brings the block scales into the layout's range, and whether the scale is rounded up until the
/// block peak is covered.
#[derive(Clone, Copy)]
struct Scale {
    layout: FloatLayout,
    block: usize,
    global: bool,
    cover: bool,
}

/// A split is the element layout and, when the elements share a scale, that scale.
#[derive(Clone, Copy)]
struct Split {
    element: FloatLayout,
    scale: Option<Scale>,
}

fn float_layout(exponent_bits: u8, mantissa_bits: u8, specials: Specials, sign: SignEncoding) -> FloatLayout {
    FloatLayout {
        exponent_bits,
        mantissa_bits,
        specials,
        bias_override: None,
        sign,
        nan: NanPolicy::IEEE,
    }
}

/// Narrow elements carry no inf/nan codes; a byte-wide element spends its top codes on them the way
/// the byte-wide OCP and IEEE-style layouts do (a wide exponent saturates, a narrow one has one NaN).
fn element_specials(exponent_bits: u8, mantissa_bits: u8) -> Specials {
    match (1 + exponent_bits + mantissa_bits, exponent_bits) {
        (..=7, _) => Specials::None,
        (_, 5..) => Specials::SaturateOnOverflow,
        _ => Specials::NanOnly,
    }
}

fn element_layout(exponent_bits: u8, mantissa_bits: u8) -> FloatLayout {
    float_layout(
        exponent_bits,
        mantissa_bits,
        element_specials(exponent_bits, mantissa_bits),
        SignEncoding::Magnitude,
    )
}

fn shared_exponent_scale() -> FloatLayout {
    float_layout(8, 0, Specials::NanOnly, SignEncoding::Absent)
}

fn small_float_scale() -> FloatLayout {
    float_layout(4, 3, Specials::NanOnly, SignEncoding::Magnitude)
}

fn half_scale() -> FloatLayout {
    float_layout(5, 10, Specials::InfinityOnOverflow, SignEncoding::Magnitude)
}

/// The scale forms the sweep crosses with every element split.
fn scale_forms() -> Vec<Option<Scale>> {
    let mut forms = vec![None];
    forms.push(Some(Scale { layout: half_scale(), block: 32, global: false, cover: false }));
    for block in [16, 32] {
        for cover in [false, true] {
            forms.push(Some(Scale { layout: shared_exponent_scale(), block, global: false, cover }));
        }
        forms.push(Some(Scale { layout: small_float_scale(), block, global: true, cover: false }));
    }
    forms
}

fn all_splits() -> Vec<Split> {
    let forms = scale_forms();
    ELEMENT_SPLITS
        .iter()
        .flat_map(|&(exponent_bits, mantissa_bits)| {
            let element = element_layout(exponent_bits, mantissa_bits);
            forms.iter().map(move |scale| Split { element, scale: *scale })
        })
        .collect()
}

fn spelled(layout: &FloatLayout) -> String {
    format!("E{}M{}", layout.exponent_bits, layout.mantissa_bits)
}

impl Split {
    fn label(&self) -> String {
        match &self.scale {
            None => format!("{} unscaled", spelled(&self.element)),
            Some(scale) => format!(
                "{} / {} per{}{}{}",
                spelled(&self.element),
                if scale.layout.sign == SignEncoding::Absent && scale.layout.mantissa_bits == 0 {
                    spelled(&scale.layout)
                } else {
                    format!("{}{}", spelled(&scale.layout), if scale.layout.exponent_bits == 5 { "(f16)" } else { "" })
                },
                scale.block,
                if scale.global { " +tensor" } else { "" },
                if scale.cover { " cover" } else { "" },
            ),
        }
    }

    fn bits_per_element(&self) -> f64 {
        let scale_bits = self
            .scale
            .map_or(0.0, |scale| f64::from(scale.layout.width()) / scale.block as f64);
        f64::from(self.element.width()) + scale_bits
    }
}

/// The accumulators of one split over one window, in f64.
#[derive(Default)]
struct Errors {
    signal: f64,
    noise: f64,
    max_rel: f64,
    max_abs: f64,
    peak: f64,
    saturated: usize,
    flushed: usize,
    count: usize,
}

impl Errors {
    fn push(&mut self, value: f32, reconstructed: f32, clipped: bool) {
        let (original, rebuilt) = (f64::from(value), f64::from(reconstructed));
        let error = (original - rebuilt).abs();
        self.signal += original * original;
        self.noise += error * error;
        self.max_abs = self.max_abs.max(error);
        self.peak = self.peak.max(original.abs());
        if value != 0.0 {
            self.max_rel = self.max_rel.max(error / original.abs());
            self.flushed += usize::from(reconstructed == 0.0);
        }
        self.saturated += usize::from(clipped);
        self.count += 1;
    }

    fn row(&self, prefix: &str, label: &str, bits: f64) -> String {
        let sqnr = if self.noise == 0.0 { f64::INFINITY } else { 10.0 * (self.signal / self.noise).log10() };
        let count = self.count as f64;
        format!(
            "{prefix}\t{label}\t{bits:.3}\t{sqnr:.2}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5}\t{:.5}",
            (self.noise / self.signal).sqrt(),
            self.max_rel,
            self.max_abs / self.peak,
            self.saturated as f64 / count,
            self.flushed as f64 / count,
        )
    }
}

fn scalar_codec(layout: &FloatLayout) -> Result<FormatCodec, Box<dyn Error>> {
    FormatCodec::scalar(layout)
        .filter(FormatCodec::can_encode)
        .ok_or_else(|| format!("no host scalar encode for {}", spelled(layout)).into())
}

/// `values` rounded into `layout` and widened back, through the codec the workspace ships.
fn round_through(layout: &FloatLayout, values: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
    let codec = scalar_codec(layout)?;
    let mut bytes = vec![0u8; values.len() * codec.block_bytes()];
    codec.encode(values, &mut bytes)?;
    let mut widened = vec![0f32; values.len()];
    decode_blocks(&NumericFormat::Scalar(*layout), &bytes, &mut widened)?;
    Ok(widened)
}

fn block_peaks(values: &[f32], block: usize) -> Vec<f32> {
    values
        .chunks_exact(block)
        .map(|chunk| chunk.iter().fold(0.0f32, |peak, value| peak.max(value.abs())))
        .collect()
}

/// One decoded scale per block, plus the per-tensor factor that multiplies all of them.
fn block_scales(scale: &Scale, values: &[f32], grid_max: f32) -> Result<(Vec<f32>, f32), Box<dyn Error>> {
    let peaks = block_peaks(values, scale.block);
    let tensor_peak = peaks.iter().fold(0.0f32, |peak, value| peak.max(*value));
    let factor = if scale.global { tensor_peak / (grid_max * scale.layout.max_normal()) } else { 1.0 };
    let mut raw: Vec<f32> = peaks.iter().map(|peak| peak / grid_max / factor).collect();
    let mut decoded = round_through(&scale.layout, &raw)?;
    if scale.cover {
        let growth = 1.0 + 0.5f32.powi(i32::from(scale.layout.mantissa_bits) + 1);
        for _ in 0..PEAK_GROWTH_PASSES {
            let mut grew = false;
            for ((raw_scale, decoded_scale), peak) in raw.iter_mut().zip(&decoded).zip(&peaks) {
                if decoded_scale * factor * grid_max < *peak {
                    *raw_scale *= growth;
                    grew = true;
                }
            }
            if !grew {
                break;
            }
            decoded = round_through(&scale.layout, &raw)?;
        }
    }
    Ok((decoded, factor))
}

/// The reconstruction of `values` under `split`, with the per-element clipped flags.
fn reconstruct(split: &Split, values: &[f32]) -> Result<(Vec<f32>, Vec<bool>), Box<dyn Error>> {
    let grid_max = split.element.max_normal();
    let scales: Vec<f32> = match &split.scale {
        None => vec![1.0; values.len()],
        Some(scale) => {
            let (decoded, factor) = block_scales(scale, values, grid_max)?;
            decoded.iter().flat_map(|decoded_scale| std::iter::repeat_n(decoded_scale * factor, scale.block)).collect()
        }
    };
    let scaled: Vec<f32> = values
        .iter()
        .zip(&scales)
        .map(|(value, scale)| if *scale == 0.0 { 0.0 } else { value / scale })
        .collect();
    let clipped = scaled.iter().map(|value| value.abs() > grid_max).collect();
    let widened = round_through(&split.element, &scaled)?;
    let rebuilt = widened.iter().zip(&scales).map(|(value, scale)| value * scale).collect();
    Ok((rebuilt, clipped))
}

fn measure(split: &Split, values: &[f32]) -> Result<Errors, Box<dyn Error>> {
    let (rebuilt, clipped) = reconstruct(split, values)?;
    let mut errors = Errors::default();
    for ((value, rebuilt_value), clip) in values.iter().zip(&rebuilt).zip(&clipped) {
        errors.push(*value, *rebuilt_value, *clip);
    }
    Ok(errors)
}

fn map_window(mapped: &Mmap, parsed: &ParsedGguf, window: &Window) -> Result<(GgmlType, Vec<f32>), Box<dyn Error>> {
    let tensor = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == window.tensor)
        .ok_or_else(|| format!("tensor {} not in the file", window.tensor))?;
    let range = parsed.tensor_data_range(tensor, mapped.len() as u64).map_err(|error| format!("{error:?}"))?;
    let shape = tensor.ggml_type.wire_shape();
    let width = tensor.dims[0] as usize;
    let row_bytes = width / shape.block_elements as usize * shape.block_bytes as usize;
    let start = range.start as usize + window.first_row * row_bytes;
    let bytes = mapped
        .get(start..start + window.rows * row_bytes)
        .ok_or("window runs past the file")?;
    let mut values = vec![0f32; window.rows * width];
    dequantize(tensor.ggml_type, bytes, &mut values).map_err(|error| format!("{error:?}"))?;
    Ok((tensor.ggml_type, values))
}

fn requantized(kind: GgmlType, quantize: Quantizer, values: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
    let shape = kind.wire_shape();
    let mut bytes = vec![0u8; values.len() / shape.block_elements as usize * shape.block_bytes as usize];
    quantize(values, &mut bytes).map_err(|error| format!("{error:?}"))?;
    let mut rebuilt = vec![0f32; values.len()];
    dequantize(kind, &bytes, &mut rebuilt).map_err(|error| format!("{error:?}"))?;
    Ok(rebuilt)
}

fn stored_bits(kind: GgmlType) -> f64 {
    let shape = kind.wire_shape();
    shape.block_bytes as f64 * 8.0 / shape.block_elements as f64
}

fn baseline_rows(prefix: &str, own: GgmlType, values: &[f32]) -> Result<Vec<String>, Box<dyn Error>> {
    let candidates: [(GgmlType, Quantizer); 7] = [
        (GgmlType::Q4_0, q4_0::quantize),
        (GgmlType::Q8_0, q8_0::quantize),
        (GgmlType::Q4_K, q4_k::quantize),
        (GgmlType::Q5_K, q5_k::quantize),
        (GgmlType::Q6_K, q6_k::quantize),
        (GgmlType::F16, f16::quantize),
        (GgmlType::Bf16, bf16::quantize),
    ];
    let mut rows = Vec::new();
    let mut identical = Errors::default();
    values.iter().for_each(|value| identical.push(*value, *value, false));
    let own_bits = stored_bits(own);
    rows.push(identical.row(prefix, &format!("gguf own {:?} (stored values, by construction)", own), own_bits));
    for (kind, quantize) in candidates {
        let rebuilt = requantized(kind, quantize, values)?;
        let mut errors = Errors::default();
        values.iter().zip(&rebuilt).for_each(|(value, rebuilt_value)| errors.push(*value, *rebuilt_value, false));
        let bits = stored_bits(kind);
        let tag = if kind == own { " (own type, re-encoded)" } else { " (re-encoded)" };
        rows.push(errors.row(prefix, &format!("gguf {kind:?}{tag}"), bits));
    }
    Ok(rows)
}

fn accuracy(mapped: &Mmap, parsed: &ParsedGguf) -> Result<(), Box<dyn Error>> {
    println!("role\ttensor\tgguf_type\telements\tsplit\tbits_per_element\tsqnr_db\trel_rms\tmax_rel\tmax_abs_over_peak\tsaturated_frac\tflushed_frac");
    let splits = all_splits();
    for window in &WINDOWS {
        let (kind, values) = map_window(mapped, parsed, window)?;
        let prefix = format!("{}\t{}\t{:?}\t{}", window.role, window.tensor, kind, values.len());
        for row in baseline_rows(&prefix, kind, &values)? {
            println!("{row}");
        }
        for split in &splits {
            let errors = measure(split, &values)?;
            println!("{}", errors.row(&prefix, &split.label(), split.bits_per_element()));
        }
    }
    Ok(())
}

const NO_DIGIT: Digit = Digit { radix: 1, stride_bits: 0 };

/// The description of a scaled split as a block layout: the scale at byte 0, then the element codes
/// packed end to end.
fn block_description(split: &Split) -> Option<BlockLayout> {
    let scale = split.scale?;
    let width = split.element.width();
    let scale_bytes = scale.layout.width().div_ceil(8) as u16;
    let code_bytes = (scale.block as u16 * u16::from(width)).div_ceil(8);
    let unused = Plane { bits: 1, base_bits: 0, digits: [NO_DIGIT; 4] };
    Some(BlockLayout {
        elements: scale.block as u16,
        block_bytes: scale_bytes + code_bytes,
        element: Coded {
            code: split.element,
            low: Plane {
                bits: width,
                base_bits: u32::from(scale_bytes) * 8,
                digits: [Digit { radix: scale.block as u16, stride_bits: u16::from(width) }, NO_DIGIT, NO_DIGIT, NO_DIGIT],
            },
            high: None,
            bias: 0,
            packed_radix: 0,
            regions: [proxima_numeric::block_layout::CodeRegion { start: 0, elements: 0, source: unused, digit_axis: 0 }; 3],
            region_count: 0,
        },
        multiplier: 1,
        scale: Field { layout: scale.layout, byte_offset: 0, pieces: None },
        min: None,
        min_multiplier: 1,
        sub: None,
        scale_runs: None,
    })
}

/// Packs `values` block by block the way `layout` says, from the host quantizer's own scales and
/// codes, so decoding the bytes through the description must give the host reconstruction.
fn pack_by_description(split: &Split, layout: &BlockLayout, values: &[f32]) -> Result<(Vec<u8>, Vec<f32>), Box<dyn Error>> {
    let scale = split.scale.ok_or("a packed block needs a scale")?;
    let grid_max = split.element.max_normal();
    let (scales, _) = block_scales(&Scale { global: false, ..scale }, values, grid_max)?;
    let scale_codec = scalar_codec(&scale.layout)?;
    let element_codec = scalar_codec(&split.element)?;
    let mut packed = vec![0u8; values.len() / scale.block * layout.block_bytes as usize];
    let mut expected = Vec::with_capacity(values.len());
    for (block_index, block) in values.chunks_exact(scale.block).enumerate() {
        let base = block_index * layout.block_bytes as usize;
        let decoded_scale = scales[block_index];
        scale_codec.encode(&[decoded_scale], &mut packed[base..base + scale.layout.width().div_ceil(8) as usize])?;
        let scaled: Vec<f32> = block.iter().map(|value| if decoded_scale == 0.0 { 0.0 } else { value / decoded_scale }).collect();
        let mut codes = vec![0u8; scaled.len()];
        element_codec.encode(&scaled, &mut codes)?;
        for (index, code) in codes.iter().enumerate() {
            let address = layout.element.low.address(index) + base * 8;
            for bit in 0..usize::from(split.element.width()) {
                packed[(address + bit) / 8] |= ((code >> bit) & 1) << ((address + bit) % 8);
            }
        }
        let widened = round_through(&split.element, &scaled)?;
        expected.extend(widened.iter().map(|value| value * decoded_scale));
    }
    Ok((packed, expected))
}

fn support(mapped: &Mmap, parsed: &ParsedGguf) -> Result<(), Box<dyn Error>> {
    println!("split\telement_scalar_codec\telement_can_encode\tblock_described\tdescription_decodes_host_bytes\tblock_table_codec\tblock_codec_can_encode\tencode_rule\tgeneric_encode_blocks");
    let (_, values) = map_window(mapped, parsed, &WINDOWS[0])?;
    for split in all_splits().iter().filter(|split| split.scale.is_none_or(|scale| !scale.cover)) {
        let element = FormatCodec::scalar(&split.element);
        let element_encode = element.as_ref().is_some_and(FormatCodec::can_encode);
        let Some(layout) = block_description(split) else {
            println!("{}\t{}\t{}\tn/a (no block)\tn/a\tn/a\tn/a\tn/a\tn/a", split.label(), element.is_some(), element_encode);
            continue;
        };
        let scale = split.scale.ok_or("scaled split")?;
        let window = &values[..scale.block * 64];
        let (packed, expected) = pack_by_description(split, &layout, window)?;
        let mut decoded = vec![0f32; window.len()];
        let decode_outcome = match decode_blocks(&NumericFormat::Block(layout), &packed, &mut decoded) {
            Ok(_) => {
                let worst = decoded.iter().zip(&expected).fold(0.0f32, |worst, (got, want)| worst.max((got - want).abs()));
                format!("yes (max |decode - host| = {worst:e} over {} values)", decoded.len())
            }
            Err(error) => format!("no: {error}"),
        };
        let table = FormatCodec::block(&layout);
        let mut out = vec![0u8; packed.len()];
        let generic = match encode_blocks(&NumericFormat::Block(layout), window, &mut out) {
            Ok(blocks) => format!("yes ({blocks} blocks)"),
            Err(error) => format!("no: {error}"),
        };
        println!(
            "{}\t{}\t{}\tyes ({} B / {} elements)\t{}\t{}\t{}\t{}\t{}",
            split.label(),
            element.is_some(),
            element_encode,
            layout.block_bytes,
            layout.elements,
            decode_outcome,
            table.is_some(),
            table.as_ref().is_some_and(FormatCodec::can_encode),
            layout.encode_rule().is_some(),
            generic,
        );
    }
    Ok(())
}

fn read_f32_le(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{path}: {} bytes is not a multiple of 4", bytes.len()).into());
    }
    let (words, _) = bytes.as_chunks::<4>();
    Ok(words.iter().map(|word| f32::from_le_bytes(*word)).collect())
}

fn raw(path: &str, role: &str, label: &str) -> Result<(), Box<dyn Error>> {
    let values = read_f32_le(path)?;
    println!("role\ttensor\tgguf_type\telements\tsplit\tbits_per_element\tsqnr_db\trel_rms\tmax_rel\tmax_abs_over_peak\tsaturated_frac\tflushed_frac");
    let prefix = format!("{role}\t{label}\traw\t{}", values.len());
    for split in &all_splits() {
        let errors = measure(split, &values)?;
        println!("{}", errors.row(&prefix, &split.label(), split.bits_per_element()));
    }
    Ok(())
}

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn unit_open(state: &mut u64) -> f64 {
    ((xorshift64(state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Box-Muller over a fixed-seed xorshift: the same N(0,1) file on every run.
fn normal_values(count: usize) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut values = Vec::with_capacity(count + 1);
    while values.len() < count {
        let radius = (-2.0 * unit_open(&mut state).ln()).sqrt();
        let angle = std::f64::consts::TAU * unit_open(&mut state);
        values.push((radius * angle.cos()) as f32);
        values.push((radius * angle.sin()) as f32);
    }
    values.truncate(count);
    values
}

fn gen_normal(path: &str, count: usize) -> Result<(), Box<dyn Error>> {
    let bytes: Vec<u8> = normal_values(count).iter().flat_map(|value| value.to_le_bytes()).collect();
    fs::write(path, bytes)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: narrow_float_split_sweep <gguf> accuracy|support | raw <file.f32> <role> [<tensor-label>] | gen-normal <out.f32> [<count>]")?;
    if path == "raw" {
        let file_path = args.next().ok_or("usage: narrow_float_split_sweep raw <file.f32> <role> [<tensor-label>]")?;
        let role = args.next().ok_or("usage: narrow_float_split_sweep raw <file.f32> <role> [<tensor-label>]")?;
        let label = args.next().unwrap_or_else(|| file_path.clone());
        return raw(&file_path, &role, &label);
    }
    if path == "gen-normal" {
        let out = args.next().ok_or("usage: narrow_float_split_sweep gen-normal <out.f32> [<count>]")?;
        let count = args.next().map_or(Ok(1_000_000), |text| text.parse())?;
        return gen_normal(&out, count);
    }
    let mode = args.next().ok_or("usage: narrow_float_split_sweep <gguf> accuracy|support")?;
    let file = File::open(&path)?;
    // the mapping is read-only and the file is not modified while this process runs
    let mapped = unsafe { Mmap::map(&file)? };
    let parsed = parse_complete(&mapped).map_err(|error| format!("{error:?}"))?;
    match mode.as_str() {
        "accuracy" => accuracy(&mapped, &parsed),
        "support" => support(&mapped, &parsed),
        other => Err(format!("unknown mode {other}").into()),
    }
}
