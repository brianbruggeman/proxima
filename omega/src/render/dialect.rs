//! The spelling a target language gives the integer steps a layout renders.
//!
//! A dialect only spells tokens: scalar and word types, literals, casts, bit reinterpretation,
//! intrinsics, a function head, a declaration, a select and a branch. Which steps a layout takes is
//! [`crate::render::load`] and friends; they are generic over `D: Dialect`, so a dialect is a type
//! parameter and never a trait object.

use alloc::borrow::Cow;
use alloc::string::String;

use super::hardware::HardwareScalar;

/// An operation every dialect spells with its own intrinsic name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intrinsic {
    LeadingZeros,
    RoundEven,
    Clamp,
    Min,
    Abs,
}

/// The target-language spelling of the render steps.
pub trait Dialect {
    /// The type name of a hardware scalar, or `None` when the dialect does not spell that row.
    fn scalar(&self, scalar: HardwareScalar) -> Option<Cow<'static, str>>;
    /// A literal of `scalar` from its `digits`.
    fn literal(&self, scalar: HardwareScalar, digits: &str) -> String;
    /// The unsigned integer type that is `bytes` wide, or `None` when the dialect has none.
    fn unsigned(&self, bytes: u8) -> Option<Cow<'static, str>>;
    /// The signed integer type that is `bytes` wide, or `None` when the dialect has none.
    fn signed(&self, bytes: u8) -> Option<Cow<'static, str>>;
    /// The suffix of an unsigned integer literal.
    fn unsigned_suffix(&self) -> &'static str;
    /// Convert `value` to `ty`.
    fn cast(&self, ty: &str, value: &str) -> String;
    /// Reinterpret the bits of `value` as `ty`.
    fn bitcast(&self, ty: &str, value: &str) -> String;
    /// Call `op` on `args`.
    fn intrinsic(&self, op: Intrinsic, args: &[&str]) -> String;
    /// The head of a device function, without its body.
    fn function(&self, returns: &str, name: &str, parameters: &str) -> String;
    /// One function parameter of type `ty` named `name`.
    fn parameter(&self, ty: &str, name: &str) -> String;
    /// A read-only byte-buffer parameter named `name`, or `None` when the dialect cannot spell one.
    fn byte_buffer(&self, name: &str) -> Option<String>;
    /// The byte at `index` of the byte buffer `buffer`.
    fn byte_at(&self, buffer: &str, index: &str) -> String;
    /// A local declaration statement.
    fn declare(&self, ty: &str, name: &str, value: &str) -> String;
    /// A conditional value.
    fn select(&self, condition: &str, then: &str, otherwise: &str) -> String;
    /// The opening of a conditional block.
    fn branch(&self, condition: &str) -> String;
}
