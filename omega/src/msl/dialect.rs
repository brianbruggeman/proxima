//! Metal Shading Language spelling of the backend-neutral render steps.
//!
//! Teaching pointer: [`crate::render::Dialect`] is the trait; this is its first implementation and
//! spells tokens only. The scalar type names live in `layout_render::SCALAR_SPELLINGS`, beside the
//! hardware types that use them. Which steps a layout takes is `crate::render::{load, store,
//! unpack}`.

use alloc::format;
use alloc::string::String;

use super::layout_render::SCALAR_SPELLINGS;
use crate::render::{Dialect, HardwareScalar, Intrinsic};

/// The Metal dialect.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Msl;

impl Dialect for Msl {
    fn scalar(&self, scalar: HardwareScalar) -> &'static str {
        SCALAR_SPELLINGS[scalar.index()].0
    }

    fn literal(&self, scalar: HardwareScalar, digits: &str) -> String {
        match SCALAR_SPELLINGS[scalar.index()].1 {
            Some(suffix) => format!("{digits}{suffix}"),
            None => format!("({}){digits}f", self.scalar(scalar)),
        }
    }

    fn unsigned(&self, bytes: u8) -> &'static str {
        match bytes {
            1 => "uchar",
            2 => "ushort",
            4 => "uint",
            _ => "ulong",
        }
    }

    fn signed(&self, bytes: u8) -> &'static str {
        match bytes {
            1 => "char",
            2 => "short",
            4 => "int",
            _ => "long",
        }
    }

    fn unsigned_suffix(&self) -> &'static str {
        "u"
    }

    fn cast(&self, ty: &str, value: &str) -> String {
        format!("({ty}){value}")
    }

    fn bitcast(&self, ty: &str, value: &str) -> String {
        format!("as_type<{ty}>({value})")
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

    fn byte_pointer(&self, name: &str) -> String {
        format!("device const uchar *{name}")
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
