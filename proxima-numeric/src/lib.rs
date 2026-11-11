//! `proxima-numeric` is the one place this workspace defines, describes or converts a number.
//!
//! Nothing here is a re-export and nothing outside this crate declares a numeric type: a format
//! that another crate needs, that crate depends on this one to name.
//!
//! Three representations, one per kind of storage:
//!
//! - [`FloatLayout`] — one number per value, as exponent width, mantissa width, special-value
//!   policy, bias and sign encoding. This workspace's `llvm::fltSemantics`. The arithmetic over
//!   it is [`float_layout_codec`].
//! - [`BlockLayout`] — a run of codes under a shared scale, with the exact bit planes, sub-block
//!   scales and mixed-radix codes that a block format needs.
//! - [`LevelFormat`] — a run of codes whose value is an explicit level set rather than the code's
//!   own layout: ternary, IQ4_NL's trained ladder, and the IQ magnitude codebooks.
//!
//! [`NumericFormat`] unites the three; [`decode::decode_blocks`] is the one place they become numbers;
//! [`encode::encode_blocks`] is the inverse; [`Numeric`] is the value that carries a format so one
//! call converts across any two of them; [`convert`] is where a value narrows into one with a named
//! rounding and overflow policy; [`DType`], [`NumericPolicy`] and [`Numerics`] are the numeric
//! decisions a tensor operation and a serving plan carry.
//!
//! Teaching pointer: start at [`NumericFormat::ALL`] to see every format this workspace knows,
//! then [`Numeric::decode`] and [`Numeric::to`] for the conversion surface. The per-format routines
//! in `proxima-gguf::quant::*` are the incumbents this crate is checked against.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
// The wire format constants keep the sidecar's own variant spellings (`Q4K`, `Iq1S`, `Mxfp4`,
// `Float16`): renaming them to upper case would rewrite 2600 call sites and every on-disk tag
// spelling for no gain.
#![expect(
    non_upper_case_globals,
    reason = "the format constants keep the wire enum's variant spellings by design"
)]

#[cfg(test)]
extern crate std;

pub mod block_codec;
pub mod block_layout;
/// The historical name for [`crate::NumericFormat`], kept so an existing call site reads
/// `Codec::Q4_0` while there is still exactly one type and one constant table.
pub mod codec {
    pub use crate::format::NumericFormat as Codec;
}
pub mod convert;
pub mod decode;
pub mod dtype;
pub mod encode;
pub mod error;
pub mod float_layout;
pub mod float_layout_codec;
pub mod format;
pub mod numeric;
pub mod numerics;
pub mod scalar_codec;
pub mod tables;
pub mod value;

pub use block_codec::BlockCodec;
pub use block_layout::{BlockLayout, Coded, Field, Plane};
pub use convert::{Overflow, Rounding, narrow_to_format, narrow_to_layout};
pub use decode::{decode_block, decode_blocks, decode_scalar, encode_scalar};
pub use dtype::DType;
pub use encode::{encode_blocks, reach};
pub use error::NumericError;
pub use float_layout::{FloatLayout, SignEncoding, Specials};
pub use format::{LevelFormat, LevelGeometry, LevelTable, NumericFormat};
pub use numeric::{NumericPolicy, NumericRewrite, admit};
pub use numerics::Numerics;
pub use scalar_codec::ScalarCodec;
pub use value::Numeric;
