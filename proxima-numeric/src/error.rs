//! Every way a numeric conversion, decode or rewrite admission can fail.

use thiserror::Error;

use crate::numeric::{NumericPolicy, NumericRewrite};

/// A numeric operation refused rather than silently producing a wrong number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum NumericError {
    #[error("format is {width} bits wide; the shared scalar codec executes at most {max} bits")]
    WidthUnsupported { width: u8, max: u8 },
    #[error(
        "layout e{exponent_bits}m{mantissa_bits} is outside the scalar codec's domain: exponent \
         bits at most {max_exponent_bits}, mantissa bits at most {max_mantissa_bits}"
    )]
    LayoutUnsupported {
        exponent_bits: u8,
        mantissa_bits: u8,
        max_exponent_bits: u8,
        max_mantissa_bits: u8,
    },
    #[error("input has {found} bytes; {format} needs a multiple of {block_bytes}")]
    InputNotBlockMultiple {
        format: &'static str,
        found: usize,
        block_bytes: usize,
    },
    #[error("output has {found} values; decoding needs {expected}")]
    OutputSizeMismatch { found: usize, expected: usize },
    #[error("block is {found} bytes; {format} needs {expected}")]
    BlockSizeMismatch {
        format: &'static str,
        found: usize,
        expected: usize,
    },
    #[error("no decoder for {format} in this crate: its codebook lands with its format frontend")]
    NoDecoder { format: &'static str },
    #[error("code {code} is outside the {format} level table")]
    LevelOutOfTable { format: &'static str, code: u32 },
    #[error("{format} geometry cannot address its own block: {reason}")]
    InvalidGeometry {
        format: &'static str,
        reason: &'static str,
    },
    #[error("numeric policy {granted:?} does not grant {rewrite:?}, which needs {required:?}")]
    PolicyTooStrict {
        rewrite: NumericRewrite,
        required: NumericPolicy,
        granted: NumericPolicy,
    },
}
