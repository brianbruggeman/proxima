//! Load, store and flat unpack of a [`FloatLayout`], spelled by a [`Dialect`].
//!
//! Teaching pointer: `proxima_numeric::float_layout_codec::{decode, encode}` is the CPU algorithm
//! and these functions render the same integer steps as target-language text. A layout whose
//! fields equal a hardware scalar ([`HardwareScalar::of`]) renders through that scalar's
//! conversion; every other layout renders as field arithmetic whose only inputs are the numeric
//! constants of the struct and of the `wide` scalar the kernel computes in. Every constant the
//! wide scalar contributes (fraction width, infinity code, quiet bit, absolute-value mask) comes
//! from `wide.layout()`, so a different wide scalar is a different argument and not a code edit.

use alloc::borrow::Cow;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_numeric::NumericFormat;
use proxima_numeric::decode_scalar;
use proxima_numeric::float_layout::{FloatLayout, NanNarrow, NanWiden, SignEncoding, Specials};

use super::dialect::{Dialect, Intrinsic};
use super::hardware::HardwareScalar;
use super::refusal::Refusal;

const fn exponent_ones(layout: &FloatLayout) -> u32 {
    (1u32 << layout.exponent_bits) - 1
}

const fn mantissa_mask(layout: &FloatLayout) -> u32 {
    ((1u64 << layout.mantissa_bits) - 1) as u32
}

const fn sign_shift(layout: &FloatLayout) -> u32 {
    layout.exponent_bits as u32 + layout.mantissa_bits as u32
}

/// The sign term a store or-s into its code: the codec drops the sign of a layout that has none,
/// so the term is zero there and a negative value stores its magnitude.
fn stored_sign(layout: &FloatLayout, top: u32, suffix: &str) -> String {
    match layout.sign {
        SignEncoding::Absent => format!("0{suffix}"),
        SignEncoding::Magnitude | SignEncoding::TwosComplement => {
            format!("(bits >> {top}{suffix}) << {}{suffix}", sign_shift(layout))
        }
    }
}

const fn infinity_code(layout: &FloatLayout) -> u32 {
    exponent_ones(layout) << layout.mantissa_bits
}

const fn max_normal_code(layout: &FloatLayout) -> u32 {
    if matches!(layout.specials, Specials::NanOnly) {
        // the NaN is the all-ones code, so the largest finite one is the code below it, whatever the
        // mantissa width (a layout with no mantissa has the exponent byte alone)
        return (infinity_code(layout) | mantissa_mask(layout)) - 1;
    }
    let top = match layout.specials {
        Specials::None | Specials::NanOnly => infinity_code(layout),
        Specials::SaturateOnOverflow | Specials::InfinityOnOverflow => {
            infinity_code(layout) - (1u32 << layout.mantissa_bits)
        }
    };
    top | mantissa_mask(layout)
}

const fn overflow_code(layout: &FloatLayout) -> u32 {
    match layout.specials {
        Specials::InfinityOnOverflow => infinity_code(layout),
        Specials::None | Specials::NanOnly | Specials::SaturateOnOverflow => {
            max_normal_code(layout)
        }
    }
}

const fn quiet_bit(layout: &FloatLayout) -> u32 {
    (1u32 << layout.mantissa_bits) >> 1
}

const fn reserves_all_ones(layout: &FloatLayout) -> bool {
    matches!(
        layout.specials,
        Specials::SaturateOnOverflow | Specials::InfinityOnOverflow
    )
}

fn unsigned_of<D: Dialect>(dialect: &D, bytes: u8) -> Result<Cow<'static, str>, Refusal> {
    dialect
        .unsigned(bytes)
        .ok_or(Refusal::NoIntegerOfWidth { bytes })
}

fn scalar_of<D: Dialect>(dialect: &D, row: HardwareScalar) -> Result<Cow<'static, str>, Refusal> {
    dialect.scalar(row).ok_or(Refusal::NoWideScalar)
}

/// The dialect plus the wide scalar the steps compute in, with every token and constant the wide
/// scalar contributes resolved once.
struct Tokens<'dialect, D: Dialect> {
    dialect: &'dialect D,
    wide: HardwareScalar,
    layout: FloatLayout,
    ty: Cow<'static, str>,
    word: Cow<'static, str>,
    suffix: &'static str,
}

impl<'dialect, D: Dialect> Tokens<'dialect, D> {
    fn new(dialect: &'dialect D, wide: HardwareScalar) -> Result<Self, Refusal> {
        Ok(Self {
            dialect,
            wide,
            layout: wide.layout(),
            ty: dialect.scalar(wide).ok_or(Refusal::NoWideScalar)?,
            word: unsigned_of(dialect, wide.bytes())?,
            suffix: dialect.unsigned_suffix(),
        })
    }

    fn signed(&self, bytes: u8) -> Result<Cow<'static, str>, Refusal> {
        self.dialect
            .signed(bytes)
            .ok_or(Refusal::NoIntegerOfWidth { bytes })
    }

    fn to_bits(&self, row: HardwareScalar, value: &str) -> Result<String, Refusal> {
        self.dialect
            .to_bits(row, value)
            .ok_or(Refusal::NoIntegerOfWidth { bytes: row.bytes() })
    }

    fn to_scalar(&self, row: HardwareScalar, code: &str) -> Result<String, Refusal> {
        self.dialect
            .to_scalar(row, code)
            .ok_or(Refusal::NoWideScalar)
    }

    fn literal(&self, digits: &str) -> String {
        self.dialect.literal(self.wide, digits)
    }

    fn fraction_bits(&self) -> u32 {
        u32::from(self.layout.mantissa_bits)
    }

    fn word_bits(&self) -> u32 {
        u32::from(self.layout.width())
    }

    fn top_bit(&self) -> u32 {
        sign_shift(&self.layout)
    }

    fn abs_mask(&self) -> u32 {
        (1u32 << self.top_bit()) - 1
    }

    fn admit(&self, layout: &FloatLayout) -> Result<(), Refusal> {
        if u32::from(layout.width()) > self.word_bits() {
            return Err(Refusal::CodeWiderThanWord {
                width: layout.width(),
            });
        }
        Ok(())
    }

    fn word_head(&self, name: &str, parameters: &str) -> String {
        self.dialect.function(&self.word, name, parameters)
    }

    fn wide_head(&self, name: &str, parameters: &str) -> String {
        self.dialect.function(&self.ty, name, parameters)
    }
}

fn integer_scales<D: Dialect>(
    tokens: &Tokens<'_, D>,
    layout: &FloatLayout,
) -> Result<(String, String), Refusal> {
    let wide = tokens.layout;
    let exponent = 1 - i32::from(layout.bias()) - i32::from(layout.mantissa_bits);
    let ceiling =
        i32::try_from(exponent_ones(&wide)).map_err(|_| Refusal::ScaleUnrepresentable)? - 1;
    let biased = (exponent + i32::from(wide.bias())).clamp(1, ceiling);
    let field = u32::try_from(biased).map_err(|_| Refusal::ScaleUnrepresentable)?;
    let bits = field << wide.mantissa_bits;
    let value = decode_scalar(&NumericFormat::Scalar(wide), &bits.to_le_bytes())
        .map_err(|_| Refusal::ScaleUnrepresentable)?;
    Ok((format!("{value:e}"), format!("{:e}", 1.0 / value)))
}

fn integer_load<D: Dialect>(
    tokens: &Tokens<'_, D>,
    layout: &FloatLayout,
    name: &str,
) -> Result<String, Refusal> {
    let dialect = tokens.dialect;
    let suffix = tokens.suffix;
    let mantissa_bits = layout.mantissa_bits;
    let mask = (1u64 << mantissa_bits) - 1;
    let (scale, _) = integer_scales(tokens, layout)?;
    // 64-bit integer arithmetic here slowed the q8_0 k-cache load on metal, so the wide type is used only when 32 bits cannot hold the signed difference; the q8_0 arm of cached_attention_decode_split_times_a_binary16_k_cache_against_a_q4_0_k_cache (omega/tests/cached_attention_decode_split_parity.rs, feature instrument) prints that load's per-step gpu time, and a_rendered_wide_integer_load_equals_the_host_decode_bit_for_bit_on_boundary_codes (layout_render_tests.rs) pins the long arm
    let fits = u32::from(mantissa_bits) + 2 <= tokens.word_bits();
    let mantissa = dialect.cast(&tokens.ty, "mantissa");
    let value = match layout.sign {
        SignEncoding::Absent => mantissa,
        SignEncoding::Magnitude => dialect.select(
            &format!("((code >> {mantissa_bits}{suffix}) & 1{suffix}) == 0{suffix}"),
            &mantissa,
            &format!("-{mantissa}"),
        ),
        SignEncoding::TwosComplement => {
            let signed = tokens.signed(if fits {
                tokens.wide.bytes()
            } else {
                tokens.wide.bytes() * 2
            })?;
            let low = dialect.cast(&signed, "mantissa");
            let high = dialect.cast(
                &signed,
                &format!(
                    "(((code >> {mantissa_bits}{suffix}) & 1{suffix}) << {mantissa_bits}{suffix})"
                ),
            );
            dialect.cast(&tokens.ty, &format!("({low} - {high})"))
        }
    };
    let head = tokens.wide_head(name, &dialect.parameter(&tokens.word, "code"));
    let declaration = dialect.declare(
        &tokens.word,
        "mantissa",
        &format!("code & {mask:#x}{suffix}"),
    );
    let scale = tokens.literal(&scale);
    Ok(format!(
        "\n{head} {{\n    {declaration}\n    return ({value}) * {scale};\n}}\n"
    ))
}

fn integer_store<D: Dialect>(
    tokens: &Tokens<'_, D>,
    layout: &FloatLayout,
    name: &str,
) -> Result<String, Refusal> {
    let dialect = tokens.dialect;
    let suffix = tokens.suffix;
    let word = tokens.word.as_ref();
    let mantissa_bits = layout.mantissa_bits;
    let mask = (1u64 << mantissa_bits) - 1;
    let (_, unscale) = integer_scales(tokens, layout)?;
    let scaled = dialect.declare(
        &tokens.ty,
        "scaled",
        &format!("value * {}", tokens.literal(&unscale)),
    );
    let zero = tokens.literal("0.0");
    // a bit test, not `scaled != scaled`: the metal compiler's fast math assumes no nan and folds the self-compare away
    let is_nan = format!(
        "({} & {:#010x}{suffix}) > {:#010x}{suffix}",
        tokens.to_bits(tokens.wide, "value")?,
        tokens.abs_mask(),
        infinity_code(&tokens.layout)
    );
    let rounded = dialect.declare(
        &tokens.ty,
        "rounded",
        &dialect.select(
            &is_nan,
            &zero,
            &dialect.intrinsic(Intrinsic::RoundEven, &["scaled"]),
        ),
    );
    let digits = u32::from(tokens.layout.mantissa_bits) + 1;
    let limit = tokens.literal(&format!("{}.0", largest_float_within(mask, digits)));
    let saturated = format!("{mask:#x}{suffix}");
    let body = match layout.sign {
        SignEncoding::Absent => {
            let clamped = dialect.intrinsic(
                Intrinsic::Clamp,
                &["rounded", zero.as_str(), limit.as_str()],
            );
            let code = dialect.select(
                &format!("rounded > {limit}"),
                &saturated,
                &dialect.cast(word, &clamped),
            );
            format!("    {rounded}\n    return {code};")
        }
        SignEncoding::Magnitude => {
            let extent = dialect.declare(
                &tokens.ty,
                "extent",
                &dialect.intrinsic(Intrinsic::Abs, &["rounded"]),
            );
            let minimum = dialect.intrinsic(Intrinsic::Min, &["extent", limit.as_str()]);
            let code = dialect.select(
                &format!("extent > {limit}"),
                &saturated,
                &dialect.cast(word, &minimum),
            );
            let magnitude = dialect.declare(word, "magnitude", &code);
            let sign_bit = tokens.to_bits(tokens.wide, "scaled")?;
            let sign_position = tokens.top_bit();
            let sign = format!(
                "({})",
                dialect.select(
                    &is_nan,
                    &format!("0{suffix}"),
                    &format!(
                        "(({sign_bit} >> {sign_position}{suffix}) << {mantissa_bits}{suffix})"
                    ),
                )
            );
            format!("    {rounded}\n    {extent}\n    {magnitude}\n    return magnitude | {sign};")
        }
        SignEncoding::TwosComplement => {
            let negative = tokens.literal(&format!("-{}.0", 1u64 << mantissa_bits));
            let clamped = dialect.intrinsic(
                Intrinsic::Clamp,
                &["rounded", negative.as_str(), limit.as_str()],
            );
            let signed = tokens.signed(tokens.wide.bytes())?;
            let declaration = dialect.declare(&signed, "clamped", &dialect.cast(&signed, &clamped));
            let full_mask = (1u64 << (mantissa_bits + 1)) - 1;
            let code = dialect.select(
                &format!("rounded > {limit}"),
                &saturated,
                &format!(
                    "({} & {full_mask:#x}{suffix})",
                    dialect.cast(word, "clamped")
                ),
            );
            format!("    {rounded}\n    {declaration}\n    return {code};")
        }
    };
    let head = tokens.word_head(name, &dialect.parameter(&tokens.ty, "value"));
    Ok(format!("\n{head} {{\n    {scaled}\n{body}\n}}\n"))
}

/// The largest float of `digits` significant bits that is no greater than `mask`: a 32-bit
/// code's `2^32 - 1` is not a float, and the float above it would admit a value the integer
/// conversion cannot take.
fn largest_float_within(mask: u64, digits: u32) -> u64 {
    let width = u64::BITS - mask.leading_zeros();
    let dropped = width.saturating_sub(digits);
    (mask >> dropped) << dropped
}

/// Renders the function `name` that takes the `layout.width()`-bit `code` and returns its value in
/// the `wide` scalar.
///
/// # Errors
/// [`Refusal::ScaleUnrepresentable`] when an integer layout's scale is not a value of `wide`.
#[must_use = "the refusal names why the layout could not be rendered"]
pub fn load<D: Dialect>(
    dialect: &D,
    wide: HardwareScalar,
    layout: FloatLayout,
    name: &str,
) -> Result<String, Refusal> {
    let tokens = Tokens::new(dialect, wide)?;
    tokens.admit(&layout)?;
    if !layout.is_float() {
        return integer_load(&tokens, &layout, name);
    }
    let value = match HardwareScalar::of(dialect, &layout) {
        Some(row) if row == wide => tokens.to_scalar(wide, "code")?,
        Some(row) => {
            let storage = dialect.cast(&unsigned_of(dialect, row.bytes())?, "code");
            dialect.cast(&tokens.ty, &tokens.to_scalar(row, &storage)?)
        }
        None => return field_load(&tokens, &layout, name),
    };
    let head = tokens.wide_head(name, &dialect.parameter(&tokens.word, "code"));
    Ok(format!("\n{head} {{\n    return {value};\n}}\n"))
}

fn field_load<D: Dialect>(
    tokens: &Tokens<'_, D>,
    layout: &FloatLayout,
    name: &str,
) -> Result<String, Refusal> {
    let dialect = tokens.dialect;
    let signed_type = tokens.signed(tokens.wide.bytes())?;
    let (word, suffix, signed) = (tokens.word.as_ref(), tokens.suffix, signed_type.as_ref());
    let mantissa_bits = layout.mantissa_bits;
    let field_mask = mantissa_mask(layout);
    let field_ones = exponent_ones(layout);
    let field_sign = sign_shift(layout);
    let bias = layout.bias();
    let fraction = tokens.fraction_bits();
    let fraction_shift = fraction - u32::from(mantissa_bits);
    let implicit = 1u64 << mantissa_bits;
    let reserved = field_load_reserved(tokens, layout, field_ones, fraction_shift)?;
    // no mantissa field means no subnormal range: field 0 is itself the lowest power of two
    // (`FloatLayout::lowest_exponent`, OCP MX v1.0 Table 7), never zero
    let (zero_exponent_significand, zero_exponent_scale) = if mantissa_bits == 0 {
        (format!("1{suffix}"), layout.lowest_exponent().to_string())
    } else {
        (
            String::from("mantissa"),
            format!("(1 - ({bias}) - {mantissa_bits})"),
        )
    };
    let wide_bias = i32::from(tokens.layout.bias());
    let wide_ones = exponent_ones(&tokens.layout);
    let wide_infinity = infinity_code(&tokens.layout);
    let wide_fraction_mask = mantissa_mask(&tokens.layout);
    let subnormal = wide_bias + i32::from(tokens.layout.mantissa_bits) - 1;
    let top = tokens.top_bit();
    let word_bits = tokens.word_bits();
    let leading = dialect.cast(
        signed,
        &dialect.intrinsic(Intrinsic::LeadingZeros, &["significand"]),
    );
    let lead = dialect.declare(signed, "lead", &format!("{top} - {leading}"));
    let biased_word = dialect.cast(word, "biased");
    let fraction_lead = dialect.cast(word, &format!("({fraction} - lead)"));
    let shift_word = dialect.cast(word, "shift");
    let negative_shift_word = dialect.cast(word, "(-shift)");
    let scaled_parameters = [
        dialect.parameter(word, "significand"),
        dialect.parameter(signed, "scale"),
    ]
    .join(", ");
    let scaled_head = tokens.word_head(&format!("{name}_scaled_bits"), &scaled_parameters);
    let load_head = tokens.wide_head(name, &dialect.parameter(word, "code"));
    let mantissa_declaration =
        dialect.declare(word, "mantissa", &format!("code & {field_mask:#x}{suffix}"));
    let exponent_declaration = dialect.declare(
        word,
        "exponent",
        &format!("(code >> {mantissa_bits}{suffix}) & {field_ones:#x}{suffix}"),
    );
    let sign_declaration = dialect.declare(
        word,
        "sign",
        &format!("((code >> {field_sign}{suffix}) & 1{suffix}) << {top}{suffix}"),
    );
    let significand_declaration = dialect.declare(
        word,
        "significand",
        &dialect.select(
            &format!("exponent == 0{suffix}"),
            &zero_exponent_significand,
            &format!("({implicit:#x}{suffix} | mantissa)"),
        ),
    );
    let scale_declaration = dialect.declare(
        signed,
        "scale",
        &dialect.select(
            &format!("exponent == 0{suffix}"),
            &zero_exponent_scale,
            &format!(
                "({} - ({bias}) - {mantissa_bits})",
                dialect.cast(signed, "exponent")
            ),
        ),
    );
    let biased_declaration =
        dialect.declare(signed, "biased", &format!("lead + scale + {wide_bias}"));
    let shift_declaration = dialect.declare(signed, "shift", &format!("scale + {subnormal}"));
    let underflow = dialect.select(
        &format!("-shift >= {word_bits}"),
        &format!("0{suffix}"),
        &format!("significand >> {negative_shift_word}"),
    );
    let result = tokens.to_scalar(
        tokens.wide,
        &format!("sign | {name}_scaled_bits(significand, scale)"),
    )?;
    Ok(format!(
        "
{scaled_head} {{
    if (significand == 0{suffix}) {{
        return 0{suffix};
    }}
    {lead}
    {biased_declaration}
    if (biased >= {wide_ones}) {{
        return {wide_infinity:#010x}{suffix};
    }}
    if (biased >= 1) {{
        return ({biased_word} << {fraction}{suffix}) | ((significand << {fraction_lead}) & {wide_fraction_mask:#x}{suffix});
    }}
    {shift_declaration}
    if (shift >= 0) {{
        return significand << {shift_word};
    }}
    return {underflow};
}}

{load_head} {{
    {mantissa_declaration}
    {exponent_declaration}
    {sign_declaration}
{reserved}    {significand_declaration}
    {scale_declaration}
    return {result};
}}
"
    ))
}

fn field_load_reserved<D: Dialect>(
    tokens: &Tokens<'_, D>,
    layout: &FloatLayout,
    field_ones: u32,
    fraction_shift: u32,
) -> Result<String, Refusal> {
    let suffix = tokens.suffix;
    let infinity = infinity_code(&tokens.layout);
    let quiet = quiet_bit(&tokens.layout);
    let payload = match layout.nan.widen {
        NanWiden::Quiet => format!(" | (mantissa << {fraction_shift}{suffix})"),
        NanWiden::Canonical => String::new(),
    };
    if layout.specials == Specials::NanOnly {
        let value = tokens.to_scalar(
            tokens.wide,
            &format!("sign | {infinity:#010x}{suffix} | {quiet:#x}{suffix}{payload}"),
        )?;
        return Ok(format!(
            "    if (exponent == {field_ones:#x}{suffix} && mantissa == {:#x}{suffix}) {{\n        return {value};\n    }}\n",
            mantissa_mask(layout)
        ));
    }
    if !reserves_all_ones(layout) {
        return Ok(String::new());
    }
    let nan_bits = tokens.dialect.select(
        &format!("mantissa == 0{suffix}"),
        &format!("0{suffix}"),
        &format!("({quiet:#x}{suffix}{payload})"),
    );
    let value = tokens.to_scalar(
        tokens.wide,
        &format!("sign | {infinity:#010x}{suffix} | ({nan_bits})"),
    )?;
    Ok(format!(
        "    if (exponent == {field_ones:#x}{suffix}) {{\n        return {value};\n    }}\n"
    ))
}

/// Renders the function `name` that takes a `wide` value and returns the `layout.width()`-bit code
/// nearest it, ties to the even code, overflow and NaN under the layout's `Specials`.
///
/// # Errors
/// [`Refusal::ScaleUnrepresentable`] when an integer layout's scale is not a value of `wide`.
#[must_use = "the refusal names why the layout could not be rendered"]
pub fn store<D: Dialect>(
    dialect: &D,
    wide: HardwareScalar,
    layout: FloatLayout,
    name: &str,
) -> Result<String, Refusal> {
    let tokens = Tokens::new(dialect, wide)?;
    tokens.admit(&layout)?;
    if !layout.is_float() {
        return integer_store(&tokens, &layout, name);
    }
    let value = match HardwareScalar::of(dialect, &layout) {
        Some(row) if row == wide => tokens.to_bits(wide, "value")?,
        Some(row) => {
            let narrow = dialect.cast(&scalar_of(dialect, row)?, "value");
            dialect.cast(&tokens.word, &tokens.to_bits(row, &narrow)?)
        }
        None => return field_store(&tokens, &layout, name),
    };
    let head = tokens.word_head(name, &dialect.parameter(&tokens.ty, "value"));
    Ok(format!("\n{head} {{\n    return {value};\n}}\n"))
}

fn field_store_nan<D: Dialect>(tokens: &Tokens<'_, D>, layout: &FloatLayout) -> String {
    let suffix = tokens.suffix;
    if layout.specials == Specials::NanOnly {
        return format!(
            "        return sign | {:#x}{suffix} | {:#x}{suffix};\n",
            infinity_code(layout),
            mantissa_mask(layout)
        );
    }
    if !reserves_all_ones(layout) {
        return format!("        return {:#x}{suffix};\n", max_normal_code(layout));
    }
    let payload = match layout.nan.narrow {
        NanNarrow::Preserve => format!(
            "((bits & {:#x}{suffix}) >> {}{suffix}) | ",
            mantissa_mask(&tokens.layout),
            tokens.fraction_bits() - u32::from(layout.mantissa_bits)
        ),
        NanNarrow::Canonical => String::new(),
    };
    format!(
        "        return sign | {:#x}{suffix} | ({payload}{:#x}{suffix});\n",
        infinity_code(layout),
        quiet_bit(layout)
    )
}

/// The statements that send a NaN `value` to the layout's own NaN code, the one step a hardware
/// conversion cannot be trusted with because it emits a single canonical NaN.
pub fn store_nan_branch<D: Dialect>(
    dialect: &D,
    wide: HardwareScalar,
    layout: &FloatLayout,
) -> Result<String, Refusal> {
    let tokens = Tokens::new(dialect, wide)?;
    let suffix = tokens.suffix;
    let nan = field_store_nan(&tokens, layout);
    let bits = dialect.declare(&tokens.word, "bits", &tokens.to_bits(tokens.wide, "value")?);
    let sign = dialect.declare(
        &tokens.word,
        "sign",
        &stored_sign(layout, tokens.top_bit(), suffix),
    );
    let branch = dialect.branch(&format!(
        "(bits & {:#x}{suffix}) > {:#010x}{suffix}",
        tokens.abs_mask(),
        infinity_code(&tokens.layout)
    ));
    Ok(format!(
        "    {bits}\n    {sign}\n    {branch}\n{nan}    }}\n"
    ))
}

fn field_store<D: Dialect>(
    tokens: &Tokens<'_, D>,
    layout: &FloatLayout,
    name: &str,
) -> Result<String, Refusal> {
    let dialect = tokens.dialect;
    let signed_type = tokens.signed(tokens.wide.bytes())?;
    let (word, suffix, signed) = (tokens.word.as_ref(), tokens.suffix, signed_type.as_ref());
    let mantissa_bits = layout.mantissa_bits;
    let bias = layout.bias();
    let field_ones = exponent_ones(layout);
    let overflow = overflow_code(layout);
    let fraction = tokens.fraction_bits();
    let value_shift = fraction - u32::from(mantissa_bits);
    let shift_limit = fraction + 1;
    let nan = field_store_nan(tokens, layout);
    let wide_bias = tokens.layout.bias();
    let wide_ones = exponent_ones(&tokens.layout);
    let wide_infinity = infinity_code(&tokens.layout);
    let wide_fraction_mask = mantissa_mask(&tokens.layout);
    let implicit = 1u64 << fraction;
    let top = tokens.top_bit();
    let abs_mask = tokens.abs_mask();
    let round_parameters = [
        dialect.parameter(word, "value"),
        dialect.parameter(word, "shift"),
    ]
    .join(", ");
    let round_head = tokens.word_head(&format!("{name}_round_shift"), &round_parameters);
    let store_head = tokens.word_head(name, &dialect.parameter(&tokens.ty, "value"));
    let quotient = dialect.declare(word, "quotient", "value >> shift");
    let remainder = dialect.declare(
        word,
        "remainder",
        &format!("value & ((1{suffix} << shift) - 1{suffix})"),
    );
    let halfway = dialect.declare(
        word,
        "halfway",
        &format!("1{suffix} << (shift - 1{suffix})"),
    );
    let bits = dialect.declare(word, "bits", &tokens.to_bits(tokens.wide, "value")?);
    let sign = dialect.declare(word, "sign", &stored_sign(layout, top, suffix));
    let abs_bits = dialect.declare(word, "abs_bits", &format!("bits & {abs_mask:#x}{suffix}"));
    let nan_branch = dialect.branch(&format!("abs_bits > {wide_infinity:#010x}{suffix}"));
    let all_ones_branch = dialect.branch(&format!(
        "(abs_bits >> {fraction}{suffix}) == {wide_ones:#x}{suffix}"
    ));
    let exponent_field = dialect.declare(
        word,
        "exponent_field",
        &format!("abs_bits >> {fraction}{suffix}"),
    );
    let significand = dialect.declare(
        word,
        "significand",
        &dialect.select(
            &format!("exponent_field == 0{suffix}"),
            "abs_bits",
            &format!("((abs_bits & {wide_fraction_mask:#x}{suffix}) | {implicit:#x}{suffix})"),
        ),
    );
    let source_exponent = dialect.declare(
        signed,
        "source_exponent",
        &dialect.select(
            &format!("exponent_field == 0{suffix}"),
            "1",
            &dialect.cast(signed, "exponent_field"),
        ),
    );
    let target = dialect.declare(
        signed,
        "target_exponent",
        &format!("source_exponent - {wide_bias} + ({bias})"),
    );
    let overflow_branch = dialect.branch(&format!("target_exponent > {field_ones}"));
    let base_exponent = dialect.declare(
        word,
        "base_exponent",
        &dialect.select(
            "target_exponent < 1",
            &format!("0{suffix}"),
            &dialect.cast(word, "(target_exponent - 1)"),
        ),
    );
    let extra_shift = dialect.declare(
        word,
        "extra_shift",
        &dialect.select(
            "target_exponent < 1",
            &dialect.cast(word, "(1 - target_exponent)"),
            &format!("0{suffix}"),
        ),
    );
    let code = dialect.declare(
        word,
        "code",
        &format!("(base_exponent << {mantissa_bits}{suffix}) + {name}_round_shift(significand, {value_shift}{suffix} + extra_shift)"),
    );
    let rounded = dialect.select(
        &format!(
            "(remainder > halfway || (remainder == halfway && (quotient & 1{suffix}) == 1{suffix}))"
        ),
        &format!("quotient + 1{suffix}"),
        "quotient",
    );
    let limit = format!("{overflow:#x}{suffix}");
    let bounded = dialect.intrinsic(Intrinsic::Min, &["code", limit.as_str()]);
    Ok(format!(
        "
{round_head} {{
    if (shift == 0{suffix}) {{
        return value;
    }}
    if (shift > {shift_limit}{suffix}) {{
        return 0{suffix};
    }}
    {quotient}
    {remainder}
    {halfway}
    return {rounded};
}}

{store_head} {{
    {bits}
    {sign}
    {abs_bits}
    {nan_branch}
{nan}    }}
    {all_ones_branch}
        return sign | {overflow:#x}{suffix};
    }}
    {exponent_field}
    {significand}
    {source_exponent}
    {target}
    {overflow_branch}
        return sign | {overflow:#x}{suffix};
    }}
    {base_exponent}
    {extra_shift}
    {code}
    return sign | {bounded};
}}
"
    ))
}

/// Renders the flat element accessor `name(payload, index)`: element `index` of a densely packed
/// array of `layout`, low bits first, in the `wide` scalar.
///
/// # Errors
/// [`Refusal::NoFlatAddress`] when the width neither fills whole bytes nor divides a byte and is
/// too wide to straddle two bytes, and whatever [`load`] refuses.
#[must_use = "the refusal names why the layout has no flat accessor"]
pub fn unpack<D: Dialect>(
    dialect: &D,
    wide: HardwareScalar,
    layout: FloatLayout,
    name: &str,
) -> Result<String, Refusal> {
    let tokens = Tokens::new(dialect, wide)?;
    tokens.admit(&layout)?;
    let suffix = tokens.suffix;
    let word = tokens.word.as_ref();
    let width = u32::from(layout.width());
    let value_name = format!("{name}_value");
    let code = if width % 8 == 0 {
        let bytes = width / 8;
        let gathered = (0..bytes)
            .map(|byte| {
                let byte_read = dialect.byte_at("payload", &format!("offset + {byte}{suffix}"));
                let read = dialect.cast(word, &byte_read);
                format!("({read} << {}{suffix})", byte * 8)
            })
            .collect::<Vec<_>>()
            .join(" | ");
        let offset = dialect.declare(word, "offset", &format!("index * {bytes}{suffix}"));
        let code = dialect.declare(word, "code", &gathered);
        format!("    {offset}\n    {code}\n")
    } else if 8 % width == 0 {
        let mask = (1u32 << width) - 1;
        let bit = dialect.declare(word, "bit", &format!("index * {width}{suffix}"));
        let byte_read = dialect.byte_at("payload", &format!("bit / 8{suffix}"));
        let read = dialect.cast(word, &byte_read);
        let code = dialect.declare(
            word,
            "code",
            &format!("({read} >> (bit % 8{suffix})) & {mask:#x}{suffix}"),
        );
        format!("    {bit}\n    {code}\n")
    } else if width <= 16 {
        straddling_unpack(&tokens, width)
    } else {
        return Err(Refusal::NoFlatAddress {
            width: layout.width(),
        });
    };
    let value = load(dialect, wide, layout, &value_name)?;
    let parameters = dialect
        .byte_buffer("payload")
        .into_iter()
        .chain([dialect.parameter(word, "index")])
        .collect::<Vec<_>>()
        .join(", ");
    let head = tokens.wide_head(name, &parameters);
    Ok(format!(
        "{value}\n{head} {{\n{code}    return {value_name}(code);\n}}\n"
    ))
}

/// The statements that read a `width`-bit field at bit `index * width` of a dense low-bits-first
/// stream, joining the two or three bytes that cover it. Each trailing byte is read only when the
/// field reaches into it, so the last element never reads past the payload.
fn straddling_unpack<D: Dialect>(tokens: &Tokens<'_, D>, width: u32) -> String {
    let dialect = tokens.dialect;
    let (word, suffix) = (tokens.word.as_ref(), tokens.suffix);
    let mask = (1u32 << width) - 1;
    let spans = (7 + width).div_ceil(8);
    let joined = (1..spans)
        .map(|byte| {
            let byte_read = dialect.byte_at("payload", &format!("first + {byte}{suffix}"));
            let read = dialect.cast(word, &byte_read);
            let reaches = dialect.select(
                &format!("shift + {width}{suffix} > {}{suffix}", byte * 8),
                &format!("{read} << {}{suffix}", byte * 8),
                &format!("0{suffix}"),
            );
            format!(" | ({reaches})")
        })
        .collect::<String>();
    let bit = dialect.declare(word, "bit", &format!("index * {width}{suffix}"));
    let first = dialect.declare(word, "first", &format!("bit / 8{suffix}"));
    let shift = dialect.declare(word, "shift", &format!("bit % 8{suffix}"));
    let first_read = dialect.cast(word, &dialect.byte_at("payload", "first"));
    let joined_word = dialect.declare(word, "word", &format!("{first_read}{joined}"));
    let code = dialect.declare(
        word,
        "code",
        &format!("(word >> shift) & {mask:#x}{suffix}"),
    );
    format!("    {bit}\n    {first}\n    {shift}\n    {joined_word}\n    {code}\n")
}
