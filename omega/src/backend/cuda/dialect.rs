//! CUDA C (and HIP C++) spelling of the backend-neutral render steps.
//!
//! Teaching pointer: [`crate::render::Dialect`] is the trait; this spells tokens only. Which steps
//! a layout takes is `crate::render::{load, store, unpack}`. The widest float's name comes from
//! the one hardware table (`crate::msl::layout_render::SCALAR_SPELLINGS`), the bfloat name from
//! [`Target::bfloat_type`] and the 64-bit integer from `card_c_integer` over [`Target::word`]; one
//! dialect serves CUDA and HIP because only that name differs.

use alloc::borrow::Cow;
use alloc::format;
use alloc::string::String;

use crate::cuda::card_c_integer;
use crate::cuda::target::Target;
use crate::msl::dialect::Msl;
use crate::msl::layout_render::SCALAR_SPELLINGS;
use crate::render::{Dialect, HardwareScalar, Intrinsic};

/// The type of CUDA's 16-bit float, which both card-C languages spell the same.
const HALF_TYPE: &str = "__half";

/// What a hardware row is to the card: the widest float, the 16-bit float with the widest float's
/// exponent range (bfloat), or the narrower-exponent 16-bit float.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Wide,
    Brain,
    Half,
}

/// The CUDA / HIP dialect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cuda {
    pub target: Target,
}

impl Cuda {
    fn class(row: HardwareScalar) -> Option<Class> {
        SCALAR_SPELLINGS[row.index()]?;
        let wide = HardwareScalar::widest(&Msl)?;
        if row == wide {
            Some(Class::Wide)
        } else if row.layout().exponent_bits == wide.layout().exponent_bits {
            Some(Class::Brain)
        } else {
            Some(Class::Half)
        }
    }

    /// The type name of `row`, borrowed for the life of the program.
    #[must_use]
    pub fn token(&self, row: HardwareScalar) -> Option<&'static str> {
        match Self::class(row)? {
            Class::Wide => SCALAR_SPELLINGS[row.index()].map(|(name, _)| name),
            Class::Brain => Some(self.target.bfloat_type),
            Class::Half => Some(HALF_TYPE),
        }
    }
}

impl Dialect for Cuda {
    fn scalar(&self, scalar: HardwareScalar) -> Option<Cow<'static, str>> {
        self.token(scalar).map(Cow::Borrowed)
    }

    fn literal(&self, scalar: HardwareScalar, digits: &str) -> String {
        match (Self::class(scalar), self.token(scalar)) {
            (Some(Class::Wide), _) => {
                let suffix = SCALAR_SPELLINGS[scalar.index()].and_then(|(_, suffix)| suffix);
                format!("{digits}{}", suffix.unwrap_or("f"))
            }
            (_, Some(name)) => format!("({name}){digits}f"),
            (_, None) => format!("{digits}f"),
        }
    }

    fn unsigned(&self, bytes: u8) -> Option<Cow<'static, str>> {
        match bytes {
            1 => Some(Cow::Borrowed("unsigned char")),
            2 => Some(Cow::Borrowed("unsigned short")),
            4 => Some(Cow::Borrowed("unsigned int")),
            8 => Some(Cow::Borrowed("unsigned long long")),
            _ => None,
        }
    }

    fn signed(&self, bytes: u8) -> Option<Cow<'static, str>> {
        match bytes {
            1 => Some(Cow::Borrowed("signed char")),
            2 => Some(Cow::Borrowed("short")),
            4 => Some(Cow::Borrowed("int")),
            _ => self
                .target
                .word_layout()
                .filter(|layout| u32::from(layout.width()) == u32::from(bytes) * u8::BITS)
                .as_ref()
                .and_then(card_c_integer)
                .map(Cow::Borrowed),
        }
    }

    fn unsigned_suffix(&self) -> &'static str {
        "u"
    }

    fn cast(&self, ty: &str, value: &str) -> String {
        format!("({ty}){value}")
    }

    fn to_bits(&self, row: HardwareScalar, value: &str) -> Option<String> {
        let name = match Self::class(row)? {
            Class::Wide => "__float_as_uint",
            Class::Half => "__half_as_ushort",
            Class::Brain => "__bfloat16_as_ushort",
        };
        Some(format!("{name}({value})"))
    }

    fn to_scalar(&self, row: HardwareScalar, code: &str) -> Option<String> {
        let name = match Self::class(row)? {
            Class::Wide => "__uint_as_float",
            Class::Half => "__ushort_as_half",
            Class::Brain => "__ushort_as_bfloat16",
        };
        Some(format!("{name}({code})"))
    }

    fn intrinsic(&self, op: Intrinsic, args: &[&str]) -> String {
        if let (Intrinsic::Clamp, [value, low, high]) = (op, args) {
            return format!("min(max({value}, {low}), {high})");
        }
        let name = match op {
            Intrinsic::LeadingZeros => "__clz",
            Intrinsic::RoundEven => "rint",
            Intrinsic::Clamp => "clamp",
            Intrinsic::Min => "min",
            Intrinsic::Abs => "fabs",
        };
        format!("{name}({})", args.join(", "))
    }

    fn function(&self, returns: &str, name: &str, parameters: &str) -> String {
        format!("__device__ __forceinline__ {returns} {name}({parameters})")
    }

    fn parameter(&self, ty: &str, name: &str) -> String {
        format!("{ty} {name}")
    }

    fn byte_buffer(&self, name: &str) -> Option<String> {
        let byte = self.unsigned(1)?;
        Some(format!("const {byte} *{name}"))
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
