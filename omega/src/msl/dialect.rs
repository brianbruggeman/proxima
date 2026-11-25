//! Metal Shading Language spelling of the backend-neutral render steps.
//!
//! Teaching pointer: [`crate::render::Dialect`] is the trait; this is its first implementation and
//! spells tokens only. The scalar type names live in `layout_render::SCALAR_SPELLINGS`, beside the
//! hardware types that use them. Which steps a layout takes is `crate::render::{load, store,
//! unpack}`.

use alloc::borrow::Cow;
use alloc::format;
use alloc::string::String;

use super::layout_render::SCALAR_SPELLINGS;
use crate::render::{Dialect, HardwareScalar, Intrinsic};

/// The Metal dialect.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Msl;

impl Dialect for Msl {
    fn scalar(&self, scalar: HardwareScalar) -> Option<Cow<'static, str>> {
        SCALAR_SPELLINGS[scalar.index()].map(|(name, _)| Cow::Borrowed(name))
    }

    fn literal(&self, scalar: HardwareScalar, digits: &str) -> String {
        let spelling = SCALAR_SPELLINGS[scalar.index()];
        match spelling {
            Some((_, Some(suffix))) => format!("{digits}{suffix}"),
            Some((name, None)) => format!("({name}){digits}f"),
            None => format!("{digits}f"),
        }
    }

    fn unsigned(&self, bytes: u8) -> Option<Cow<'static, str>> {
        match bytes {
            1 => Some(Cow::Borrowed("uchar")),
            2 => Some(Cow::Borrowed("ushort")),
            4 => Some(Cow::Borrowed("uint")),
            8 => Some(Cow::Borrowed("ulong")),
            _ => None,
        }
    }

    fn signed(&self, bytes: u8) -> Option<Cow<'static, str>> {
        match bytes {
            1 => Some(Cow::Borrowed("char")),
            2 => Some(Cow::Borrowed("short")),
            4 => Some(Cow::Borrowed("int")),
            8 => Some(Cow::Borrowed("long")),
            _ => None,
        }
    }

    fn unsigned_suffix(&self) -> &'static str {
        "u"
    }

    fn cast(&self, ty: &str, value: &str) -> String {
        format!("({ty}){value}")
    }

    fn to_bits(&self, row: HardwareScalar, value: &str) -> Option<String> {
        let word = self.unsigned(row.bytes())?;
        Some(format!("as_type<{word}>({value})"))
    }

    fn to_scalar(&self, row: HardwareScalar, code: &str) -> Option<String> {
        let scalar = self.scalar(row)?;
        Some(format!("as_type<{scalar}>({code})"))
    }

    fn intrinsic(&self, op: Intrinsic, args: &[&str]) -> String {
        let name = match op {
            Intrinsic::LeadingZeros => "clz",
            Intrinsic::RoundEven => "rint",
            Intrinsic::Clamp => "clamp",
            Intrinsic::Min => "min",
            Intrinsic::Abs => "fabs",
        };
        format!("{name}({})", args.join(", "))
    }

    fn function(&self, returns: &str, name: &str, parameters: &str) -> String {
        format!("static inline {returns} {name}({parameters})")
    }

    fn parameter(&self, ty: &str, name: &str) -> String {
        format!("{ty} {name}")
    }

    fn byte_buffer(&self, name: &str) -> Option<String> {
        let byte = self.unsigned(1)?;
        Some(format!("device const {byte} *{name}"))
    }

    fn byte_at(&self, buffer: &str, index: &str) -> String {
        format!("{buffer}[{index}]")
    }

    fn declare(&self, ty: &str, name: &str, value: &str) -> String {
        format!("{ty} {name} = {value};")
    }

    fn select(&self, condition: &str, then: &str, otherwise: &str) -> String {
        format!("{condition} ? {then} : {otherwise}")
    }

    fn branch(&self, condition: &str) -> String {
        format!("if ({condition}) {{")
    }
}
