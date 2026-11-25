//! WebGPU Shading Language spelling of the backend-neutral render steps.
//!
//! Teaching pointer: [`crate::render::Dialect`] is the trait; this spells tokens only. Which steps
//! a layout takes is `crate::render::{load, store, unpack}`. WGSL has one 32-bit float and 32-bit
//! integers, so the type names are derived from the word width rather than spelled, and every row
//! of another width has no spelling here.

use alloc::borrow::Cow;
use alloc::format;
use alloc::string::String;

use crate::render::{Dialect, HardwareScalar, Intrinsic};

/// The width in bits of the one float and the integers WGSL computes in.
pub(crate) const WORD_BITS: u8 = 32;

fn derived(prefix: &str, bits: u32) -> Cow<'static, str> {
    Cow::Owned(format!("{prefix}{bits}"))
}

fn integer(prefix: &str, bytes: u8) -> Option<Cow<'static, str>> {
    let bits = u32::from(bytes) * u8::BITS;
    (bits == u32::from(WORD_BITS)).then(|| derived(prefix, bits))
}

/// The WGSL dialect.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Wgsl;

impl Dialect for Wgsl {
    fn scalar(&self, scalar: HardwareScalar) -> Option<Cow<'static, str>> {
        let width = scalar.layout().width();
        (width == WORD_BITS).then(|| derived("f", u32::from(width)))
    }

    fn literal(&self, _scalar: HardwareScalar, digits: &str) -> String {
        String::from(digits)
    }

    fn unsigned(&self, bytes: u8) -> Option<Cow<'static, str>> {
        integer("u", bytes)
    }

    fn signed(&self, bytes: u8) -> Option<Cow<'static, str>> {
        integer("i", bytes)
    }

    fn unsigned_suffix(&self) -> &'static str {
        "u"
    }

    fn cast(&self, ty: &str, value: &str) -> String {
        format!("{ty}({value})")
    }

    fn to_bits(&self, row: HardwareScalar, value: &str) -> Option<String> {
        let word = self.unsigned(row.bytes())?;
        Some(format!("bitcast<{word}>({value})"))
    }

    fn to_scalar(&self, row: HardwareScalar, code: &str) -> Option<String> {
        let scalar = self.scalar(row)?;
        Some(format!("bitcast<{scalar}>({code})"))
    }

    fn intrinsic(&self, op: Intrinsic, args: &[&str]) -> String {
        let name = match op {
            Intrinsic::LeadingZeros => "countLeadingZeros",
            Intrinsic::RoundEven => "round",
            Intrinsic::Clamp => "clamp",
            Intrinsic::Min => "min",
            Intrinsic::Abs => "abs",
        };
        format!("{name}({})", args.join(", "))
    }

    fn function(&self, returns: &str, name: &str, parameters: &str) -> String {
        format!("fn {name}({parameters}) -> {returns}")
    }

    fn parameter(&self, ty: &str, name: &str) -> String {
        format!("{name}: {ty}")
    }

    fn byte_buffer(&self, _name: &str) -> Option<String> {
        None
    }

    fn byte_at(&self, buffer: &str, index: &str) -> String {
        let word_bytes = u32::from(WORD_BITS) / u8::BITS;
        let mask = (1u32 << u8::BITS) - 1;
        let suffix = self.unsigned_suffix();
        format!(
            "(({buffer}[({index}) / {word_bytes}{suffix}] >> ((({index}) % {word_bytes}{suffix}) * {}{suffix})) & {mask:#x}{suffix})",
            u8::BITS
        )
    }

    fn declare(&self, ty: &str, name: &str, value: &str) -> String {
        format!("let {name}: {ty} = {value};")
    }

    fn select(&self, condition: &str, then: &str, otherwise: &str) -> String {
        format!("select({otherwise}, {then}, {condition})")
    }

    fn branch(&self, condition: &str) -> String {
        format!("if ({condition}) {{")
    }
}
