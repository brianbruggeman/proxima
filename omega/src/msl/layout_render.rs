//! The one MSL emitter over a [`FloatLayout`]: load (decode) and store (encode) from the fields.
//!
//! Teaching pointer: `proxima_numeric::float_layout_codec::{decode, encode}` is the CPU algorithm
//! and `crate::render` renders the same integer steps as target-language text, so the two agree code
//! for code; this module calls it with the `Msl` dialect and the widest hardware scalar. A layout
//! whose fields equal a Metal hardware type (`float`, `half`, `bfloat`) renders through that type's
//! conversion; every other layout renders as field arithmetic whose only inputs are the numeric
//! constants of the struct. Compose the pieces with [`layout_unpack_msl`] for a flat element
//! accessor, or call [`layout_load_msl`] and [`layout_store_msl`] directly when the caller owns the
//! element addressing (the scaled-block decoders in `scaled_float_render` do).

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_numeric::block_layout::{BlockLayout, CodeRegion, Coded, Field, Plane};
use proxima_numeric::float_layout::FloatLayout;
use proxima_numeric::{DType, FormatCodec, LevelFormat, LevelGeometry, LevelTable, NumericFormat};

use crate::render::{self, HARDWARE_COUNT, HardwareScalar, Refusal};

use super::dialect::Msl;

/// The Metal spelling of each row of the hardware table, in table order: the type name and, when
/// Metal gives it one, the literal suffix; a row Metal does not spell is `None`. [`Msl`] reads this
/// table.
pub(crate) const SCALAR_SPELLINGS: [Option<(&str, Option<&str>)>; HARDWARE_COUNT] = [
    Some(("float", Some("f"))),
    None,
    Some(("half", Some("h"))),
    Some(("bfloat", None)),
];

/// A Metal scalar type a [`FloatLayout`] equals field by field: its name and the bytes one
/// element occupies. The vector and `simdgroup` 8x8 spellings derive from the name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HardwareType {
    pub scalar: &'static str,
    pub bytes: u64,
    zero_suffix: Option<&'static str>,
}

impl HardwareType {
    fn from_scalar(scalar: HardwareScalar) -> Option<Self> {
        SCALAR_SPELLINGS[scalar.index()].map(|(name, zero_suffix)| Self {
            scalar: name,
            bytes: u64::from(scalar.bytes()),
            zero_suffix,
        })
    }
    /// The zero literal of this type: `0.0h` for `half`, `0.0f` for `float`, and the scalar cast
    /// of `0.0f` for a type Metal gives no literal suffix.
    #[must_use]
    pub fn zero_literal(&self) -> String {
        match self.zero_suffix {
            Some(suffix) => format!("0.0{suffix}"),
            None => format!("({})0.0f", self.scalar),
        }
    }

    /// The 4-wide vector type, e.g. `half4`.
    #[must_use]
    pub fn vector(&self) -> String {
        format!("{}4", self.scalar)
    }

    /// The `simdgroup` 8x8 matrix type, e.g. `simdgroup_half8x8`.
    #[must_use]
    pub fn matrix(&self) -> String {
        format!("simdgroup_{}8x8", self.scalar)
    }

    /// Two 4-wide vectors as one 8-element value, e.g. `matrix<half, 2, 4>` (`float2x4` for a
    /// `float` tile); a spelled-out template so narrow scalars need no typedef.
    #[must_use]
    pub fn pair_of_vectors(&self) -> String {
        format!("matrix<{}, 2, 4>", self.scalar)
    }
}

/// The Metal type whose fields equal `layout`'s, compared field by field and never by name.
/// The one table every stage type derives from ([`HardwareScalar::of`]); a layout with no row is
/// `None`, never a fallback.
#[must_use]
pub fn hardware_stage_type(layout: &FloatLayout) -> Option<HardwareType> {
    HardwareScalar::of(&Msl, layout).and_then(HardwareType::from_scalar)
}

/// The bytes one element of the hardware scalar named `scalar` (`float`, `half`, `bfloat`)
/// occupies, for a caller that holds a class by name.
#[must_use]
pub fn hardware_scalar_bytes(scalar: &str) -> Option<u64> {
    HardwareScalar::all(&Msl)
        .filter_map(HardwareType::from_scalar)
        .find(|row| row.scalar == scalar)
        .map(|row| row.bytes)
}

/// The widest hardware row: its name and bytes are the one decision for every layout the kernel
/// computes in the widest hardware float, token and buffer width alike.
#[must_use]
pub fn hardware_word() -> Option<HardwareType> {
    // why: the emitter computes every non-half/bfloat layout in the widest hardware float
    HardwareScalar::widest(&Msl).and_then(HardwareType::from_scalar)
}

/// The Metal scalar type name of [`hardware_stage_type`].
#[must_use]
pub fn hardware_type(layout: &FloatLayout) -> Option<&'static str> {
    hardware_stage_type(layout).map(|stage| stage.scalar)
}

/// The dtype a kernel can write one element of a `format` cache row as, at the cache's own width,
/// or `None` when it cannot. A scalar layout equal field by field to the Metal `half` is stored
/// and computed at 16 bits; every other scalar layout (`bfloat` included) is computed in `float`
/// (`device_scalar_token`), so its device
/// element is wider than its stored element and a row of it needs a host narrowing. A block or
/// level format has no scalar element at all.
///
/// Teaching pointer: [`hardware_type`] is the field comparison; a producer node typed with the
/// returned dtype is placed straight into a cache buffer of `format` (see `DeviceKv`).
#[must_use]
pub fn device_row_store_dtype(format: &NumericFormat) -> Option<DType> {
    let NumericFormat::Scalar(layout) = format else {
        return None;
    };
    (hardware_type(layout) == Some("half")).then_some(DType::Layout(*layout))
}

fn wide_scalar() -> Result<HardwareScalar, Refusal> {
    HardwareScalar::widest(&Msl).ok_or(Refusal::NoWideScalar)
}

// why: the public emitters return text, so a refusal becomes an `#error` line and the kernel
// fails to compile with the typed reason instead of rendering something else
fn refused(refusal: Refusal) -> String {
    format!("#error {refusal}\n")
}

/// Renders `static inline float name(uint code)`: the value of the `layout.width()`-bit `code`.
/// [`render::load`] with the [`Msl`] dialect and the widest hardware scalar.
#[must_use]
pub fn layout_load_msl(layout: FloatLayout, name: &str) -> String {
    wide_scalar()
        .and_then(|wide| render::load(&Msl, wide, layout, name))
        .unwrap_or_else(refused)
}

/// Renders `static inline uint name(float value)`: the `layout.width()`-bit code nearest `value`,
/// ties to the even code, overflow and NaN under the layout's `Specials`. [`render::store`] with
/// the [`Msl`] dialect and the widest hardware scalar.
#[must_use]
pub fn layout_store_msl(layout: FloatLayout, name: &str) -> String {
    wide_scalar()
        .and_then(|wide| render::store(&Msl, wide, layout, name))
        .unwrap_or_else(refused)
}

/// The MSL that sends a NaN `value` to the layout's own NaN code, the one step a hardware
/// conversion cannot be trusted with because it emits a single canonical NaN.
#[must_use]
pub fn field_store_nan_branch_msl(layout: &FloatLayout) -> String {
    wide_scalar()
        .and_then(|wide| render::store_nan_branch(&Msl, wide, layout))
        .unwrap_or_else(refused)
}

/// Renders `static inline float name(device const uchar *payload, uint index)`, the flat element
/// accessor: element `index` of a densely packed array of `layout`, low bits first. `None` when the
/// width neither fills whole bytes nor divides a byte, because such an element has no flat address.
#[must_use]
pub fn layout_unpack_msl(layout: FloatLayout, name: &str) -> Option<String> {
    wide_scalar()
        .and_then(|wide| render::unpack(&Msl, wide, layout, name))
        .ok()
}

/// The statements that bind `{binding}_address` to the bit address of the plane's field for
/// element `index`: the plane's digits split the index least significant first, each adding its
/// stride (`Plane::address`, as MSL).
fn plane_address_msl(plane: &Plane, index: &str, binding: &str) -> String {
    let steps = plane
        .digits
        .iter()
        .filter(|digit| digit.radix > 1)
        .map(|digit| {
            format!(
                "    {binding}_address += ({binding}_rest % {}u) * {}u;\n    {binding}_rest /= {}u;\n",
                digit.radix, digit.stride_bits, digit.radix
            )
        })
        .collect::<String>();
    format!(
        "    uint {binding}_rest = {index};\n    uint {binding}_address = {}u;\n{steps}",
        plane.base_bits
    )
}

/// The most bytes any element's field of `plane` touches: one when every field sits inside a byte,
/// two or more when a field straddles a byte boundary (a 6-bit plane at bit 6 spans two bytes).
fn plane_byte_span(plane: &Plane) -> u32 {
    let elements: usize = plane
        .digits
        .iter()
        .map(|digit| usize::from(digit.radix))
        .product();
    (0..elements)
        .map(|index| (plane.address(index) as u32 % 8 + u32::from(plane.bits)).div_ceil(8))
        .max()
        .unwrap_or(1)
}

/// Reads one field of `plane` for element `index` into `uint {binding}`. A field inside one byte is
/// one byte load shifted and masked. A field that straddles bytes joins the bytes that cover it,
/// each trailing byte read only when the field reaches into it (`shift + bits > 8 * byte`), so the
/// last field of a block never reads past the block, and no high bit of the field is dropped.
fn plane_expression(plane: &Plane, index: &str, binding: &str) -> String {
    let mask = (1u32 << plane.bits) - 1;
    let address = plane_address_msl(plane, index, binding);
    let bits = plane.bits;
    let span = plane_byte_span(plane);
    if span <= 1 {
        return format!(
            "{address}    uint {binding} = ((uint)block[{binding}_address / 8u] >> ({binding}_address % 8u)) & {mask:#x}u;\n"
        );
    }
    let joined = (1..span)
        .map(|byte| {
            format!(
                " | ({binding}_address % 8u + {bits}u > {}u ? (uint)block[{binding}_address / 8u + {byte}u] << {}u : 0u)",
                byte * 8,
                byte * 8
            )
        })
        .collect::<String>();
    format!(
        "{address}    uint {binding}_joined = (uint)block[{binding}_address / 8u]{joined};\n    uint {binding} = ({binding}_joined >> ({binding}_address % 8u)) & {mask:#x}u;\n"
    )
}

/// How a run of consecutive elements of one plane is read with word loads: every lane `r` of the
/// run sits at bit `A + r * stride_bits`, with `A` the address of the run's first element.
struct RunWords {
    stride_bits: u32,
    /// The shift `A % 8` when every run of the block has the same one, else `None` (it is then read
    /// at run time and the stride is a whole number of bytes).
    static_shift: Option<u32>,
    words: u32,
    /// `safe[w]`: word `w` lies inside the block for every run, so it may be one `uint` load.
    safe: Vec<bool>,
    /// Every run's first byte is a multiple of 4 inside a block whose size is a multiple of 4, so
    /// a block origin that is word-aligned makes every `safe` word an aligned `uint` load.
    word_aligned: bool,
}

/// Decides whether `plane`'s fields for the `run` elements starting at any multiple of `run` can
/// be read from a window of `uint` words, and how. The rule: the plane's first non-trivial digit
/// must be a multiple of `run` wide, so no run crosses a digit carry and its fields lie at the
/// constant bit stride of that digit; the run's fields must then fit in the window under one of
/// two cases. If every run starts at the same bit shift inside its first byte, each field's bit
/// position is a compile-time constant and a field that crosses a word boundary is joined from the
/// two words that hold it. If the shift varies per run, the stride must be whole bytes and every
/// field must sit inside its word at the largest shift. A plane that fits neither is read per
/// element ([`plane_expression`]).
fn run_words(plane: &Plane, layout: &BlockLayout, run: u64) -> Option<RunWords> {
    let digit = plane.digits.iter().find(|digit| digit.radix > 1)?;
    let run = u32::try_from(run).ok()?;
    if run == 0 || u32::from(digit.radix) % run != 0 {
        return None;
    }
    let stride = u32::from(digit.stride_bits);
    let bits = u32::from(plane.bits);
    let starts: Vec<u32> = (0..u32::from(layout.elements))
        .step_by(run as usize)
        .map(|slot| plane.address(slot as usize) as u32)
        .collect();
    let shifts: Vec<u32> = starts.iter().map(|address| address % 8).collect();
    let first = *shifts.first()?;
    let static_shift = shifts.iter().all(|shift| *shift == first).then_some(first);
    let max_shift = shifts.iter().copied().max()?;
    let reach = (run - 1) * stride + bits;
    if static_shift.is_none() {
        let lanes_fit =
            stride % 8 == 0 && (0..run).all(|lane| (lane * stride) % 32 + max_shift + bits <= 32);
        if !lanes_fit {
            return None;
        }
    }
    let words = (max_shift + reach).div_ceil(32);
    let safe = (0..words)
        .map(|word| {
            starts
                .iter()
                .all(|address| address / 8 + 4 * (word + 1) <= u32::from(layout.block_bytes))
        })
        .collect();
    let word_aligned = layout.block_bytes.is_multiple_of(4)
        && starts.iter().all(|address| (address / 8).is_multiple_of(4));
    Some(RunWords {
        stride_bits: stride,
        static_shift,
        words,
        safe,
        word_aligned,
    })
}

/// Statements binding `uint {binding}_codes[run]` to the plane's field for each element of the
/// run starting at `slot`: each word is one aligned `uint` load (a runtime pointer-alignment check
/// falls back to assembling the word from guarded byte reads), and each lane is a constant shift
/// and mask of its word, joined across a word boundary where its field straddles one.
fn run_words_msl(
    plane: &Plane,
    plan: &RunWords,
    layout: &BlockLayout,
    run: u64,
    slot: &str,
    binding: &str,
    aligned: bool,
) -> String {
    let mask = (1u32 << plane.bits) - 1;
    let bits = u32::from(plane.bits);
    let block_bytes = layout.block_bytes;
    let mut text = plane_address_msl(plane, slot, binding);
    text.push_str(&format!(
        "    uint {binding}_first = {binding}_address / 8u;\n    uint {binding}_shift = {binding}_address % 8u;\n    device const uchar *{binding}_bytes = block + {binding}_first;\n"
    ));
    if !aligned {
        text.push_str(&format!(
            "    bool {binding}_aligned = (((ulong){binding}_bytes) & 3ul) == 0ul;\n"
        ));
    }
    for word in 0..plan.words {
        let assembled = (0..4)
            .map(|byte| {
                let offset = 4 * word + byte;
                format!("({binding}_first + {offset}u < {block_bytes}u ? (uint){binding}_bytes[{offset}u] << {}u : 0u)", byte * 8)
            })
            .collect::<Vec<_>>()
            .join(" | ");
        let load = if plan.safe[word as usize] && aligned {
            format!("((device const uint *){binding}_bytes)[{word}u]")
        } else if plan.safe[word as usize] {
            format!(
                "{binding}_aligned ? ((device const uint *){binding}_bytes)[{word}u] : ({assembled})"
            )
        } else {
            assembled
        };
        text.push_str(&format!("    uint {binding}_w{word} = {load};\n"));
    }
    text.push_str(&format!("    uint {binding}_codes[{run}];\n"));
    if plan.static_shift.is_none() {
        text.push_str(&word_lane_masks_msl(plan, bits, run as u32, binding));
    }
    for lane in 0..run as u32 {
        let offset = lane * plan.stride_bits;
        text.push_str(&match plan.static_shift {
            Some(shift) => {
                let position = shift + offset;
                let (word, at) = (position / 32, position % 32);
                if at + bits <= 32 {
                    format!("    {binding}_codes[{lane}] = ({binding}_w{word} >> {at}u) & {mask:#x}u;\n")
                } else {
                    format!("    {binding}_codes[{lane}] = (({binding}_w{word} >> {at}u) | ({binding}_w{} << {}u)) & {mask:#x}u;\n", word + 1, 32 - at)
                }
            }
            None => format!(
                "    {binding}_codes[{lane}] = ({binding}_a{} >> {}u) & {mask:#x}u;\n",
                offset / 32,
                offset % 32
            ),
        });
    }
    text
}

// the run loop indexes `codes[]` and `out[]` per trip; only the clang spelling forces the full
// unroll that keeps them in registers (proxnum-589).
const RUN_UNROLL: &str = "_Pragma(\"clang loop unroll(full)\")";

/// The runtime-shift word form: each loaded word is shifted by the run's bit shift and masked to
/// its lanes' fields once (ggml's `(q4[i] >> 4*il) & 0x0f0f0f0f`), so a lane is then a constant
/// shift of that word instead of a shift by a runtime sum.
fn word_lane_masks_msl(plan: &RunWords, bits: u32, run: u32, binding: &str) -> String {
    let field = (1u32 << bits) - 1;
    (0..plan.words)
        .filter_map(|word| {
            let offsets = (0..run).map(|lane| lane * plan.stride_bits);
            let in_word = offsets.filter(|offset| offset / 32 == word);
            let lanes_mask = in_word.fold(0u32, |joined, offset| joined | (field << (offset % 32)));
            (lanes_mask != 0).then(|| {
                let shifted = format!("({binding}_w{word} >> {binding}_shift)");
                format!("    uint {binding}_a{word} = {shifted} & {lanes_mask:#x}u;\n")
            })
        })
        .collect()
}

fn coded_expression(coded: &Coded, index: &str, binding: &str, helper: &str) -> String {
    let (mut text, code) = coded_assembly(coded, index, binding);
    text.push_str(&format!("    float {binding} = {helper}({code});\n"));
    text
}

/// The statements that assemble `coded`'s integer code for element `index`, and the expression that names it.
pub(super) fn coded_assembly(coded: &Coded, index: &str, binding: &str) -> (String, String) {
    if coded.packed_radix > 1 && coded.region_count > 0 {
        let code_name = format!("{binding}_code");
        let mut text = format!("    uint {code_name} = 0u;\n");
        for (region_index, region) in coded
            .regions
            .iter()
            .take(usize::from(coded.region_count))
            .enumerate()
        {
            text.push_str(&radix_region_expression(
                region,
                coded.packed_radix,
                index,
                &code_name,
                region_index,
                if region_index == 0 { "if" } else { "else if" },
            ));
        }
        return (text, code_name);
    }

    let mut text = plane_expression(&coded.low, index, &format!("{binding}_low"));
    let mut code = format!("{binding}_low");
    if let Some(high) = &coded.high {
        text.push_str(&plane_expression(high, index, &format!("{binding}_high")));
        code = format!("({binding}_low | ({binding}_high << {}u))", coded.low.bits);
    }
    (text, code)
}

fn radix_region_expression(
    region: &CodeRegion,
    radix: u8,
    index: &str,
    binding: &str,
    region_index: usize,
    condition: &str,
) -> String {
    let start = u32::from(region.start);
    let end = start + u32::from(region.elements);
    let local = format!("{binding}_region_{region_index}_index");
    let packed = format!("{binding}_region_{region_index}_packed");
    let address_map = plane_expression(&region.source, &local, &packed);
    let axis = usize::from(region.digit_axis);
    let divisor = region
        .source
        .digits
        .iter()
        .take(axis)
        .fold(1u32, |product, digit| product * u32::from(digit.radix));
    let lane_radix = u32::from(region.source.digits[axis].radix);
    let storage_bits = u32::from(region.source.bits);
    let storage_mask = (1u32 << storage_bits) - 1;
    format!(
        "    {condition} ({index} >= {start}u && {index} < {end}u) {{\n        uint {local} = {index} - {start}u;\n{address_map}        uint {binding}_region_{region_index}_lane = ({local} / {divisor}u) % {lane_radix}u;\n        uint {binding}_region_{region_index}_factor = 1u;\n        for (uint power = 0u; power < {binding}_region_{region_index}_lane; ++power) {{ {binding}_region_{region_index}_factor *= {radix}u; }}\n        uint {binding}_region_{region_index}_residue = ({packed} * {binding}_region_{region_index}_factor) & {storage_mask:#x}u;\n        {binding} = ({binding}_region_{region_index}_residue * {radix}u) >> {storage_bits}u;\n    }}\n"
    )
}

fn field_expression(field: &Field, binding: &str, helper: &str) -> String {
    let offset = field.byte_offset;
    let bytes = usize::from(field.layout.width()) / 8;
    let gathered = (0..bytes)
        .map(|byte| {
            format!(
                "((uint)block[{}u] << {}u)",
                usize::from(offset) + byte,
                byte * 8
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    format!("    float {binding} = {helper}({gathered});\n")
}

/// [`field_expression`] for the block scale: the scale field of the run that holds `index`, which
/// is the layout's one scale when it declares no scale runs. Every element of a four-wide load is
/// in one run, because a run is a whole number of four-wide loads.
fn scale_expression(layout: &BlockLayout, binding: &str, helper: &str) -> String {
    let Some(runs) = layout.scale_runs else {
        return field_expression(&layout.scale, binding, helper);
    };
    let first = layout.scale.byte_offset;
    let span = layout.run_elements();
    let stride = runs.stride_bytes;
    let gathered = (0..usize::from(layout.scale.layout.width()) / 8)
        .map(|byte| {
            format!(
                "((uint)block[{first}u + (index / {span}u) * {stride}u + {byte}u] << {}u)",
                byte * 8
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    format!("    float {binding} = {helper}({gathered});\n")
}

/// The block layout a cached-attention kernel reads a cache stored as `format` through, `None`
/// when the format is not one. [`block_unpack_msl`] must render the layout whole (it has no
/// per-run scale fields), the host codec must resolve it and have an encode instantiation (the
/// device cache is filled by narrowing f32 rows with `FormatCodec::encode`, so a decode-only codec
/// would be admitted and fail at the first commit), and four consecutive elements must stay inside
/// one block so a four-wide load never straddles two. Decided from the layout's own properties,
/// never from a format name; [`block_k_refusal`] names the property that failed.
#[must_use]
pub fn readable_block_layout(format: &NumericFormat) -> Option<&BlockLayout> {
    block_k_refusal(format).ok()
}

/// The layout and the Metal storage type (`uchar`, `ushort` or `uint`) a cache of `format` is
/// addressed through when it is a scalar whose code fills whole bytes, so a four-wide load is
/// four contiguous codes and [`layout_load_msl`] decodes each. `None` for a block or level
/// format, and for a width with no flat address (a 6-bit code straddles bytes). The host codec
/// must also resolve and encode it, because the device cache is filled by encoding rows.
/// Decided from the layout's width, never from a format name.
#[must_use]
pub fn readable_scalar_cache(format: &NumericFormat) -> Option<(FloatLayout, &'static str)> {
    let NumericFormat::Scalar(layout) = format else {
        return None;
    };
    let storage = match layout.width() {
        8 => "uchar",
        16 => "ushort",
        32 => "uint",
        _ => return None,
    };
    FormatCodec::for_format(format)
        .is_some_and(|codec| codec.can_encode())
        .then_some((*layout, storage))
}

/// The float layout of a format that stores one element per packed block in a whole number of
/// bytes dividing 32 bits (8, 16 or 32), which a kernel can read as a typed or byte buffer and
/// decode one code at a time with [`layout_load_msl`]. `None` for a block or level format, a
/// sub-byte or odd-width scalar, and an integer layout (no exponent field). Decided from the
/// description alone, never from a format name: the dense GEMM's admission and its wide weight
/// stage both ask this one question.
#[must_use]
pub fn one_element_float_layout(format: &NumericFormat) -> Option<FloatLayout> {
    let NumericFormat::Scalar(layout) = format else {
        return None;
    };
    let width = layout.width();
    let whole = matches!(width, 8 | 16 | 32) && format.packed_elements() == 1;
    (whole && layout.exponent_bits > 0).then_some(*layout)
}

/// What a kernel needs to read a whole-byte scalar K/V cache: the buffer element type, the
/// per-code decode name and the helper text that defines it.
pub(super) struct ScalarCacheRead {
    /// `uchar`, `ushort` or `uint`: the element type the cache buffer is declared with.
    pub storage: &'static str,
    /// `omega_kv_scalar_<fingerprint>`: `float name(uint code)` decodes one code; `{name}_x4` decodes
    /// the four codes of one `storage` vector.
    pub load: String,
    /// The text defining `load` and `{load}_x4`, pushed once after the preamble.
    pub helper: String,
}

/// The one rendered read of a scalar cache, for every attention form: the per-code decode is
/// [`layout_load_msl`] on the cache layout, and the `_x4` line applies it to the four codes of
/// a `storage` vector (the float4 read shape; the scalar shape calls `load` directly).
/// The name carries the layout's description fingerprint, so layouts that share a geometry
/// never share a helper. `None` when [`readable_scalar_cache`] is.
#[must_use]
pub(super) fn scalar_cache_read(format: &NumericFormat) -> Option<ScalarCacheRead> {
    let (layout, storage) = readable_scalar_cache(format)?;
    let load = format!(
        "omega_kv_scalar_{}",
        crate::identity::description_fingerprint(&layout)
    );
    let element = layout_load_msl(layout, &load);
    let helper = format!(
        "{element}static inline float4 {load}_x4({storage}4 value) {{ return float4({load}((uint)value.x), {load}((uint)value.y), {load}((uint)value.z), {load}((uint)value.w)); }}\n"
    );
    Some(ScalarCacheRead {
        storage,
        load,
        helper,
    })
}

/// Whether a cached-attention kernel reads a cache stored as `format` through a rendered layout
/// helper: [`readable_block_layout`] or [`readable_scalar_cache`]. Operand admission asks this
/// beside its list of weight codecs, so a K or V cache in a described layout reaches the kernel that
/// reads it by that description and not by a name on a list.
#[must_use]
pub fn cache_read_renders(format: &NumericFormat) -> bool {
    readable_block_layout(format).is_some() || readable_scalar_cache(format).is_some()
}

/// [`readable_block_layout`] with the reason: `Err` names the property the format lacks.
///
/// # Errors
/// The property of the format that makes it unreadable as a block K.
#[must_use = "the refusal names why a format cannot be a block K"]
pub fn block_k_refusal(format: &NumericFormat) -> Result<&BlockLayout, &'static str> {
    let NumericFormat::Block(layout) = format else {
        return Err("not a block layout");
    };
    if layout.elements % 4 != 0 || layout.run_elements() % 4 != 0 {
        return Err("the layout's block or scale run is not a whole number of four-wide loads");
    }
    if layout.scale_runs.is_some() && layout.scale.layout.width() % 8 != 0 {
        return Err("a scale run field that does not fill whole bytes has no flat address");
    }
    let Some(codec) = FormatCodec::for_format(format) else {
        return Err("no host codec resolves the layout");
    };
    if !codec.can_encode() {
        return Err("the host codec has no encode, so the device cache cannot be filled");
    }
    Ok(layout)
}

/// Renders `static inline float name(device const uchar *block, uint index)`: element `index` of
/// one block of `layout`, assembled from its planes and scaled by its fields and sub-block codes.
/// The helpers `name_*` are the layout loads of the code, scale and min layouts.
#[must_use]
pub fn block_unpack_msl(layout: &BlockLayout, name: &str) -> String {
    let element = block_element_msl(layout, name);
    let run = block_run_msl(layout, name);
    format!("{element}{run}")
}

/// [`block_unpack_msl`] without the four-wide `name_run` accessor: the scalar `name(block, index)`
/// and its `name_*` load helpers only. A format whose callers read one element at a time (the
/// legacy `q4_0_element`, `q5_0_element`, `q5_1_element` accessors) renders this, so no unused
/// four-wide body enters the translation unit.
#[must_use]
pub fn block_element_msl(layout: &BlockLayout, name: &str) -> String {
    element_msl(layout, Codes::Affine, name)
}

/// Renders `static inline float {prefix}_super_element(superblock, index)`: the row-blocked matvec's
/// per-format reader, which selects the `index / elements`-th block of `block_bytes` bytes in the
/// superblock span and reads element `index % elements` through `{prefix}_element`. Both counts
/// come from `layout`, so a format whose `{prefix}_element` is rendered (`block_element_msl`,
/// `scaled_block_element_msl`) gets its reader with no per-format code.
#[must_use]
pub fn block_super_element_msl(layout: &BlockLayout, prefix: &str) -> String {
    let (elements, bytes) = (layout.elements, layout.block_bytes);
    format!(
        "
static inline float {prefix}_super_element(device const uchar *superblock, uint index) {{
    return {prefix}_element(superblock + (index / {elements}u) * {bytes}u, index % {elements}u);
}}
"
    )
}

/// The property of a level format that keeps [`level_block_element_msl`] from rendering it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("level element renderer cannot render this format: {0}")]
pub struct LevelRenderRefusal(pub &'static str);

/// Renders `static inline float name(device const uchar *block, uint index)` for a level-set
/// format: [`block_element_msl`]'s walk of the block's planes and sub-block codes, with the code
/// resolved to a value through the format's [`LevelTable`] instead of the layout's affine map.
///
/// An explicit ladder (IQ4_NL, IQ4_XS) becomes a constant table indexed by the code; a linear
/// ladder is its `zero_point + step * code`; a signed grid (IQ3_S) becomes a constant table of
/// packed magnitudes indexed by the code, a lane picked by `index % lanes`, and a sign read from the
/// sign plane. The sub-block scale is `scale_multiplier * code - bias`, as `proxima_numeric::decode`
/// computes it. Teaching pointer: this is the renderer the tiled GEMM's `tiled_decode` description
/// and the cooperative kernel's `iq4_xs_element` / `iq3_s_element` accessors both call.
///
/// # Errors
/// The level table or layout property the renderer does not cover (grids with a sign table, 64-bit
/// grids, delta grids, scale runs, a block min).
#[must_use = "the refusal names why the format has no rendered element accessor"]
pub fn level_block_element_msl(
    format: &LevelFormat,
    name: &str,
) -> Result<String, LevelRenderRefusal> {
    let layout = renderable_level_layout(format)?;
    Ok(element_msl(layout, Codes::Levels(&format.levels), name))
}

/// Renders the tiled decode contract `function(device const uchar *block, uint slot, thread half
/// *out)` for a level-set format: the `run` consecutive elements at `slot`, each the same value
/// [`level_block_element_msl`]'s accessor `name` returns for it, with the block scale and the
/// sub-block scale decoded once for the run instead of once per element. Teaching pointer: the
/// accessor stays the scalar reference (the cooperative kernel reads it per element); this is the
/// run form the tiled and expert-grouped GEMM stage weights through, so the scale arithmetic
/// leaves the per-element loop.
///
/// # Errors
/// What [`level_block_element_msl`] refuses, and a layout whose sub-blocks are narrower than
/// `run` (a run would then straddle two sub-block scales).
#[must_use = "the refusal names why the format has no rendered run decode"]
pub fn level_run_decode_msl(
    format: &LevelFormat,
    function: &str,
    name: &str,
    run: u64,
) -> Result<String, LevelRenderRefusal> {
    run_decode_msl(format, function, name, run, false)
}

/// The variant of [`level_run_decode_msl`] that loads every in-block word as one `uint` with no
/// alignment select, or `None` when the layout does not prove it: it requires a block size and
/// every run's plane byte offset to be multiples of 4 (`RunWords::word_aligned`), so the only
/// remaining fact is that the block origin is word-aligned, which the caller selects on at run time
/// (one uniform per kernel) and which a misaligned binding never satisfies. The decision reads the
/// layout description alone, never a format name.
#[must_use = "None means the layout cannot prove word alignment"]
pub fn level_run_word_decode_msl(
    format: &LevelFormat,
    function: &str,
    name: &str,
    run: u64,
) -> Option<String> {
    run_decode_msl(format, function, name, run, true).ok()
}

/// The per-element form of [`level_run_decode_msl`]: one plane walk and one level lookup per
/// element, never the word or grouped run reads. It is the reference the grouped signed-grid run
/// decode is compared against bit for bit.
///
/// # Errors
/// What [`level_run_decode_msl`] refuses.
#[doc(hidden)]
#[must_use = "the refusal names why the format has no rendered run decode"]
pub fn level_run_decode_per_element_msl(
    format: &LevelFormat,
    function: &str,
    name: &str,
    run: u64,
) -> Result<String, LevelRenderRefusal> {
    run_decode_form_msl(format, function, name, run, RunForm::PerElement)
}

#[derive(Clone, Copy, PartialEq)]
enum RunForm {
    Runs { aligned: bool },
    PerElement,
}

fn run_decode_msl(
    format: &LevelFormat,
    function: &str,
    name: &str,
    run: u64,
    aligned: bool,
) -> Result<String, LevelRenderRefusal> {
    run_decode_form_msl(format, function, name, run, RunForm::Runs { aligned })
}

fn run_decode_form_msl(
    format: &LevelFormat,
    function: &str,
    name: &str,
    run: u64,
    form: RunForm,
) -> Result<String, LevelRenderRefusal> {
    let layout = renderable_level_layout(format)?;
    let mut head = field_expression(&layout.scale, "scale", &format!("{name}_scale"));
    let mut scale = "scale";
    if let Some(sub) = &layout.sub {
        let span = u64::from(layout.elements / sub.count);
        if span % run != 0 {
            return Err(LevelRenderRefusal(
                "a decode run that straddles two sub-block scales",
            ));
        }
        head.push_str(&format!("    uint sub_block = slot / {span}u;\n"));
        head.push_str(&coded_expression(
            &sub.scale,
            "sub_block",
            "sub_scale_raw",
            &format!("{name}_sub_scale_code"),
        ));
        head.push_str(&format!(
            "    float sub_scale = {} - {}.0f;\n    float effective_scale = scale * sub_scale;\n",
            sub_scale_product(sub.scale_multiplier),
            sub.scale.bias
        ));
        scale = "effective_scale";
    }
    let RunForm::Runs { aligned } = form else {
        return Ok(per_element_run_msl(
            layout,
            &format.levels,
            (function, name),
            run,
            (&head, scale),
        ));
    };
    if let Some(words) = word_run_level_msl(
        layout,
        &format.levels,
        (function, name),
        run,
        (&head, scale),
        aligned,
    ) {
        return Ok(words);
    }
    if aligned {
        return Err(LevelRenderRefusal("no word-aligned run decode"));
    }
    Ok(per_element_run_msl(
        layout,
        &format.levels,
        (function, name),
        run,
        (&head, scale),
    ))
}

fn per_element_run_msl(
    layout: &BlockLayout,
    levels: &LevelTable,
    (function, name): (&str, &str),
    run: u64,
    (head, scale): (&str, &str),
) -> String {
    let raw = raw_level_msl(layout, Codes::Levels(levels), name);
    format!(
        "\nstatic inline void {function}(device const uchar *block, uint slot, thread half *out) {{\n{head}    for (uint run = 0u; run < {run}u; ++run) {{\n        uint index = slot + run;\n{raw}        out[run] = (half)(raw * {scale});\n    }}\n}}\n"
    )
}

/// How `count` fields of one plane, `step` elements apart, are read from a window of `nbytes`
/// bytes that starts at the first field's byte: field `i` sits at bit `shift + i * stride_bits`.
struct PlaneWindow {
    stride_bits: u32,
    nbytes: u32,
}

/// Decides whether, for every run start, the plane's fields of the `count` elements `slot`,
/// `slot + step`, ... lie at a constant bit stride and fit in a `uint` window inside the block.
/// `None` sends the caller to the per-element form.
fn plane_window(
    plane: &Plane,
    layout: &BlockLayout,
    run: u32,
    (count, step): (u32, u32),
) -> Option<PlaneWindow> {
    let stride = if count > 1 {
        plane.address(step as usize).checked_sub(plane.address(0))? as u32
    } else {
        0
    };
    let starts: Vec<usize> = (0..usize::from(layout.elements))
        .step_by(run as usize)
        .collect();
    let steady = starts.iter().all(|start| {
        (0..count).all(|field| {
            plane.address(start + (field * step) as usize)
                == plane.address(*start) + (field * stride) as usize
        })
    });
    let max_shift = starts
        .iter()
        .map(|start| plane.address(*start) as u32 % 8)
        .max()?;
    let nbytes = (max_shift + (count - 1) * stride + u32::from(plane.bits)).div_ceil(8);
    let inside = starts
        .iter()
        .all(|start| plane.address(*start) as u32 / 8 + nbytes <= u32::from(layout.block_bytes));
    (steady && inside && nbytes <= 4).then_some(PlaneWindow {
        stride_bits: stride,
        nbytes,
    })
}

/// Statements binding `uint {binding}_window` (the plane's bytes for the run, little endian) and
/// `uint {binding}_shift` (the bit position of the run's first field in it).
fn plane_window_msl(plane: &Plane, window: &PlaneWindow, binding: &str) -> String {
    let mut text = plane_address_msl(plane, "slot", binding);
    text.push_str(&format!("    device const uchar *{binding}_bytes = block + {binding}_address / 8u;\n    uint {binding}_shift = {binding}_address % 8u;\n"));
    let joined = (0..window.nbytes)
        .map(|byte| format!("((uint){binding}_bytes[{byte}u] << {}u)", byte * 8))
        .collect::<Vec<_>>()
        .join(" | ");
    text.push_str(&format!("    uint {binding}_window = {joined};\n"));
    text
}

/// The run decode of a signed grid (IQ3_S): the run's code and sign bytes are read once into `uint`
/// windows, then one grid word is fetched per `lanes` elements and its magnitude bytes are split
/// with constant shifts; the sign is XORed into the float's sign bit, so no per-element branch,
/// byte load or plane address remains. Every element is the same value `signed_grid_raw_msl` gives.
fn signed_grid_run_msl(
    layout: &BlockLayout,
    levels: &LevelTable,
    (function, name): (&str, &str),
    run: u64,
    (head, scale): (&str, &str),
) -> Option<String> {
    let LevelTable::SignedGrid {
        lanes,
        sign,
        unit_shift,
        sign_table: None,
        ..
    } = levels
    else {
        return None;
    };
    let (run, lanes) = (u32::try_from(run).ok()?, u32::from(*lanes));
    if lanes == 0 || run % lanes != 0 {
        return None;
    }
    let groups = run / lanes;
    let coded = &layout.element;
    let low_window = plane_window(&coded.low, layout, run, (groups, lanes))?;
    let high = coded.high.as_ref();
    let high_window = match high {
        Some(plane) => Some(plane_window(plane, layout, run, (groups, lanes))?),
        None => None,
    };
    let sign_window = plane_window(sign, layout, run, (run, 1))?;
    let mut body = plane_window_msl(&coded.low, &low_window, "low");
    if let (Some(plane), Some(window)) = (high, &high_window) {
        body.push_str(&plane_window_msl(plane, window, "high"));
    }
    body.push_str(&plane_window_msl(sign, &sign_window, "sign"));
    let unit = if *unit_shift == 0 {
        String::new()
    } else {
        format!(" * {:e}f", 0.5f32.powi(i32::from(*unit_shift)))
    };
    let low_mask = (1u32 << coded.low.bits) - 1;
    for group in 0..groups {
        let mut code = format!(
            "((low_window >> (low_shift + {}u)) & {low_mask:#x}u)",
            group * low_window.stride_bits
        );
        if let (Some(plane), Some(window)) = (high, &high_window) {
            let mask = (1u32 << plane.bits) - 1;
            code.push_str(&format!(
                " | (((high_window >> (high_shift + {}u)) & {mask:#x}u) << {}u)",
                group * window.stride_bits,
                coded.low.bits
            ));
        }
        body.push_str(&format!("    uint grid{group} = {name}_grid[{code}];\n"));
    }
    for element in 0..run {
        let (group, lane) = (element / lanes, element % lanes);
        body.push_str(&format!(
            "    out[{element}] = (half)(as_type<float>(as_type<uint>((float)((grid{group} >> {}u) & 0xFFu){unit}) ^ (((sign_window >> (sign_shift + {}u)) & 1u) << 31u)) * {scale});\n",
            8 * lane,
            element * sign_window.stride_bits
        ));
    }
    Some(format!(
        "\nstatic inline void {function}(device const uchar *block, uint slot, thread half *out) {{\n{head}{body}}}\n"
    ))
}

/// The word-load form of [`level_run_decode_msl`]'s body, from the planes' description alone: each
/// plane of the element code is read as `uint` words ([`run_words`]) and the run's codes are
/// shifted and masked out of them, instead of one byte load and address recompute per element.
/// `None` when the format's code is not a plain plane (or two) whose run fits [`run_words`]'s rule,
/// so the caller keeps the per-element body.
fn word_run_level_msl(
    layout: &BlockLayout,
    levels: &LevelTable,
    (function, name): (&str, &str),
    run: u64,
    (head, scale): (&str, &str),
    aligned: bool,
) -> Option<String> {
    if matches!(levels, LevelTable::SignedGrid { .. }) {
        return if aligned {
            None
        } else {
            signed_grid_run_msl(layout, levels, (function, name), run, (head, scale))
        };
    }
    if layout.element.packed_radix > 1 {
        return None;
    }
    let coded = &layout.element;
    let low_plan = run_words(&coded.low, layout, run)?;
    if aligned && !(low_plan.word_aligned && low_plan.safe.contains(&true)) {
        return None;
    }
    let mut body = run_words_msl(&coded.low, &low_plan, layout, run, "slot", "low", aligned);
    let mut code = "low_codes[run]".to_string();
    if let Some(high) = &coded.high {
        let high_plan = run_words(high, layout, run)?;
        if aligned && !high_plan.word_aligned {
            return None;
        }
        body.push_str(&run_words_msl(
            high, &high_plan, layout, run, "slot", "high", aligned,
        ));
        code = format!(
            "(low_codes[run] | (high_codes[run] << {}u))",
            coded.low.bits
        );
    }
    let (table, raw) = level_lookup_msl(levels, name, &code);
    let params = "(device const uchar *block, uint slot, thread half *out)";
    let lane = format!("        float raw = {raw};\n        out[run] = (half)(raw * {scale});\n");
    let open = format!("    {RUN_UNROLL} for (uint run = 0u; run < {run}u; ++run) {{\n");
    let (setup, close) = (format!("{head}{body}"), "    }\n}\n");
    Some(format!(
        "{table}\nstatic inline void {function}{params} {{\n{setup}{open}{lane}{close}"
    ))
}

/// The run decode's level lookup: an explicit table becomes a `constant float` array, read as
/// float like ggml's `kvalues_iq4nl_f` with no per-lookup conversion. It is defined once under a
/// guard so the guarded and word-aligned twins share it: two copies cost 2,113 vs 1,792 us on the
/// IQ4_XS expert GEMM. Every other table keeps its `_code` helper. Returns the table definition
/// (empty when none) and the lookup expression.
fn level_lookup_msl(levels: &LevelTable, name: &str, code: &str) -> (String, String) {
    match levels {
        LevelTable::Explicit(values) => {
            let floats = values.iter().map(|value| format!("{value}.0f"));
            let entries = floats.collect::<Vec<_>>().join(", ");
            let guard = format!("{name}_levels_f_defined");
            let count = values.len();
            let table = format!(
                "\n#ifndef {guard}\n#define {guard}\n\
constant float {name}_levels_f[{count}] = {{ {entries} }};\n#endif\n"
            );
            (table, format!("{name}_levels_f[{code}]"))
        }
        _ => (String::new(), format!("{name}_code({code})")),
    }
}

fn sub_scale_product(scale_multiplier: i16) -> String {
    if scale_multiplier == 1 {
        "sub_scale_raw".to_string()
    } else {
        format!("({scale_multiplier}.0f * sub_scale_raw)")
    }
}

fn renderable_level_layout(format: &LevelFormat) -> Result<&BlockLayout, LevelRenderRefusal> {
    let LevelGeometry::Packed(layout) = &format.geometry else {
        return Err(LevelRenderRefusal("codebook geometry has no plane walk"));
    };
    if layout.scale_runs.is_some() || layout.min.is_some() {
        return Err(LevelRenderRefusal("scale runs or a block min"));
    }
    if layout.element.packed_radix > 1 {
        return Err(LevelRenderRefusal("packed-radix codes"));
    }
    match &format.levels {
        LevelTable::Linear { .. } | LevelTable::Explicit(_) => {}
        LevelTable::SignedGrid {
            sign_table: None,
            lanes,
            ..
        } if *lanes > 0 => {}
        LevelTable::SignedGrid { .. } => {
            return Err(LevelRenderRefusal(
                "a signed grid whose sign is a table index",
            ));
        }
        LevelTable::SignedGrid64 { .. } | LevelTable::DeltaGrid { .. } => {
            return Err(LevelRenderRefusal("64-bit and delta grids"));
        }
    }
    Ok(layout)
}

#[derive(Clone, Copy)]
enum Codes<'a> {
    Affine,
    Levels(&'a LevelTable),
}

fn level_table_msl(levels: &LevelTable, name: &str) -> String {
    match levels {
        LevelTable::Linear {
            zero_point, step, ..
        } => format!(
            "\nstatic inline float {name}_code(uint code) {{\n    return (float)(({zero_point}) + ({step}) * (int)code);\n}}\n"
        ),
        LevelTable::Explicit(values) => {
            let entries = values
                .iter()
                .map(i8::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "\nconstant char {name}_levels[{}] = {{ {entries} }};\nstatic inline float {name}_code(uint code) {{\n    return (float){name}_levels[code];\n}}\n",
                values.len()
            )
        }
        LevelTable::SignedGrid { grid, .. } => {
            let rows = grid
                .chunks(8)
                .map(|row| {
                    row.iter()
                        .map(|entry| format!("{entry:#010x}u"))
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .collect::<Vec<_>>()
                .join(",\n    ");
            format!(
                "\nconstant uint {name}_grid[{}] = {{\n    {rows}\n}};\n",
                grid.len()
            )
        }
        LevelTable::SignedGrid64 { .. } | LevelTable::DeltaGrid { .. } => String::new(),
    }
}

/// The statements that bind `float raw` (the element's level) for `codes`.
fn raw_level_msl(layout: &BlockLayout, codes: Codes<'_>, name: &str) -> String {
    match codes {
        Codes::Levels(LevelTable::SignedGrid {
            lanes,
            sign,
            unit_shift,
            ..
        }) => signed_grid_raw_msl(&layout.element, name, *lanes, sign, *unit_shift),
        Codes::Affine | Codes::Levels(_) => {
            coded_expression(&layout.element, "index", "raw", &format!("{name}_code"))
        }
    }
}

fn signed_grid_raw_msl(
    coded: &Coded,
    name: &str,
    lanes: u8,
    sign: &Plane,
    unit_shift: u8,
) -> String {
    let (mut text, code) = coded_assembly(coded, "index", "raw_code");
    text.push_str(&format!(
        "    uint magnitude = ({name}_grid[{code}] >> (8u * (index % {lanes}u))) & 0xFFu;\n"
    ));
    text.push_str(&plane_expression(sign, "index", "negative"));
    let unit = if unit_shift == 0 {
        String::new()
    } else {
        format!(" * {:e}f", 0.5f32.powi(i32::from(unit_shift)))
    };
    text.push_str(&format!(
        "    float raw = negative != 0u ? -(float)magnitude{unit} : (float)magnitude{unit};\n"
    ));
    text
}

fn element_msl(layout: &BlockLayout, codes: Codes<'_>, name: &str) -> String {
    let mut helpers = match codes {
        Codes::Affine => layout_load_msl(layout.element.code, &format!("{name}_code")),
        Codes::Levels(levels) => level_table_msl(levels, name),
    };
    helpers.push_str(&layout_load_msl(
        layout.scale.layout,
        &format!("{name}_scale"),
    ));
    if let Some(min) = &layout.min {
        helpers.push_str(&layout_load_msl(min.layout, &format!("{name}_min")));
    }
    let mut body = raw_level_msl(layout, codes, name);
    body.push_str(&scale_expression(layout, "scale", &format!("{name}_scale")));
    let mut scale = "scale".to_string();
    if let Some(sub) = &layout.sub {
        let span = layout.elements / sub.count;
        helpers.push_str(&layout_load_msl(
            sub.scale.code,
            &format!("{name}_sub_scale_code"),
        ));
        body.push_str(&format!("    uint sub_block = index / {span}u;\n"));
        body.push_str(&coded_expression(
            &sub.scale,
            "sub_block",
            "sub_scale_raw",
            &format!("{name}_sub_scale_code"),
        ));
        body.push_str(&format!(
            "    float sub_scale = {} - {}.0f;\n    float effective_scale = scale * sub_scale;\n",
            sub_scale_product(sub.scale_multiplier),
            sub.scale.bias
        ));
        scale = "effective_scale".to_string();
    }
    match codes {
        Codes::Affine => body.push_str(&format!(
            "    float value = (({}.0f * raw) - {}.0f) * {scale};\n",
            layout.multiplier, layout.element.bias
        )),
        Codes::Levels(_) => body.push_str(&format!("    float value = raw * {scale};\n")),
    }
    if let Some(min) = &layout.min {
        body.push_str(&field_expression(min, "minimum", &format!("{name}_min")));
        let mut minimum = "minimum".to_string();
        if let Some(sub_min) = layout.sub.and_then(|sub| sub.min) {
            helpers.push_str(&layout_load_msl(
                sub_min.code,
                &format!("{name}_sub_min_code"),
            ));
            body.push_str(&coded_expression(
                &sub_min,
                "sub_block",
                "sub_min_raw",
                &format!("{name}_sub_min_code"),
            ));
            body.push_str(&format!(
                "    float effective_min = minimum * (sub_min_raw - {}.0f);\n",
                sub_min.bias
            ));
            minimum = "effective_min".to_string();
        }
        body.push_str(&format!(
            "    value = value + ({}.0f * {minimum});\n",
            layout.min_multiplier
        ));
    }
    format!(
        "{helpers}\nstatic inline float {name}(device const uchar *block, uint index) {{\n{body}    return value;\n}}\n"
    )
}

/// Renders `static inline float4 name_run(device const uchar *block, uint index)`: elements
/// `index..index + 4` of one block, with the block scale (and min) decoded once for the run
/// instead of once per element. `index` is a multiple of four, which `block_k_refusal` guarantees
/// stays inside one block. Reuses the `name_*` load helpers [`block_unpack_msl`] renders, and
/// keeps the scalar helper's arithmetic order so the two agree bit for bit.
fn block_run_msl(layout: &BlockLayout, name: &str) -> String {
    let mut body = scale_expression(layout, "scale", &format!("{name}_scale"));
    if let Some(min) = &layout.min {
        body.push_str(&field_expression(min, "minimum", &format!("{name}_min")));
    }
    for lane in 0..4 {
        body.push_str(&run_element_msl(layout, name, lane));
    }
    format!(
        "static inline float4 {name}_run(device const uchar *block, uint index) {{\n{body}    return float4(value0, value1, value2, value3);\n}}\n"
    )
}

fn run_element_msl(layout: &BlockLayout, name: &str, lane: usize) -> String {
    let at = format!("(index + {lane}u)");
    let raw = format!("raw{lane}");
    let mut text = coded_expression(&layout.element, &at, &raw, &format!("{name}_code"));
    let mut scale = "scale".to_string();
    let mut minimum = "minimum".to_string();
    if let Some(sub) = &layout.sub {
        let span = layout.elements / sub.count;
        text.push_str(&format!("    uint sub_block{lane} = {at} / {span}u;\n"));
        text.push_str(&coded_expression(
            &sub.scale,
            &format!("sub_block{lane}"),
            &format!("sub_scale_raw{lane}"),
            &format!("{name}_sub_scale_code"),
        ));
        text.push_str(&format!(
            "    float effective_scale{lane} = scale * (sub_scale_raw{lane} - {}.0f);\n",
            sub.scale.bias
        ));
        scale = format!("effective_scale{lane}");
    }
    text.push_str(&format!(
        "    float value{lane} = (({}.0f * {raw}) - {}.0f) * {scale};\n",
        layout.multiplier, layout.element.bias
    ));
    if layout.min.is_some() {
        if let Some(sub_min) = layout.sub.and_then(|sub| sub.min) {
            text.push_str(&coded_expression(
                &sub_min,
                &format!("sub_block{lane}"),
                &format!("sub_min_raw{lane}"),
                &format!("{name}_sub_min_code"),
            ));
            text.push_str(&format!(
                "    float effective_min{lane} = minimum * (sub_min_raw{lane} - {}.0f);\n",
                sub_min.bias
            ));
            minimum = format!("effective_min{lane}");
        }
        text.push_str(&format!(
            "    value{lane} = value{lane} + ({}.0f * {minimum});\n",
            layout.min_multiplier
        ));
    }
    text
}

#[cfg(test)]
mod hardware_type_tests {
    use super::HardwareType;
    use crate::msl::dialect::Msl;
    use crate::render::HardwareScalar;

    #[test]
    fn zero_literal_takes_the_suffix_column_and_casts_a_type_with_no_suffix() {
        let types = HardwareScalar::all(&Msl)
            .filter_map(HardwareType::from_scalar)
            .collect::<Vec<_>>();

        let literals = types
            .iter()
            .map(HardwareType::zero_literal)
            .collect::<Vec<_>>();

        assert_eq!(literals[..2], ["0.0f", "0.0h"]);
        assert_eq!(literals[2], format!("({})0.0f", types[2].scalar));
    }
}
