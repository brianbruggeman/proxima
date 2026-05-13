//! Parse / encode / session error types.
//!
//! All types are `Copy` and carry enough context (tag byte, field name,
//! lengths) to explain the failure from a log line alone. No allocation,
//! no `std` — `std::error::Error` impls are feature-gated.

use core::fmt;

/// Failure while decoding a wire message.
///
/// `tag` is the message-type byte of the frame being decoded; untagged
/// startup-phase messages report `tag = 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParseError {
    /// frame length field below the minimum legal for the message shape
    BadLength { tag: u8, length: i32 },
    /// message body ended before a required field
    Truncated { tag: u8 },
    /// bytes remained after the last field of the body
    TrailingBytes { tag: u8, trailing: usize },
    /// message-type byte not defined for this direction
    UnknownTag { tag: u8 },
    /// untagged initial message carried an unknown request code
    UnknownRequestCode { code: i32 },
    /// startup major protocol version is not 3
    UnsupportedProtocol { major: u16, minor: u16 },
    /// a string field was missing its NUL terminator
    MissingNul { tag: u8 },
    /// a field carried a value outside its legal domain
    InvalidValue { tag: u8, field: &'static str },
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadLength { tag, length } => {
                write!(formatter, "bad length {length} for message tag {tag:#04x}")
            }
            Self::Truncated { tag } => {
                write!(formatter, "body truncated in message tag {tag:#04x}")
            }
            Self::TrailingBytes { tag, trailing } => {
                write!(
                    formatter,
                    "{trailing} trailing bytes in message tag {tag:#04x}"
                )
            }
            Self::UnknownTag { tag } => write!(formatter, "unknown message tag {tag:#04x}"),
            Self::UnknownRequestCode { code } => {
                write!(formatter, "unknown startup request code {code}")
            }
            Self::UnsupportedProtocol { major, minor } => {
                write!(formatter, "unsupported protocol version {major}.{minor}")
            }
            Self::MissingNul { tag } => {
                write!(
                    formatter,
                    "string missing nul terminator in message tag {tag:#04x}"
                )
            }
            Self::InvalidValue { tag, field } => {
                write!(
                    formatter,
                    "invalid value for {field} in message tag {tag:#04x}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseError {}

/// Failure while encoding a wire message into a caller-owned buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// output buffer cannot hold the encoded message; `needed` is the
    /// total byte count the encode would have required
    BufferTooSmall { needed: usize },
    /// a value exceeds what its wire field can represent
    ValueTooLarge { field: &'static str },
    /// a value is outside its legal wire domain (e.g. embedded NUL in a
    /// string field)
    InvalidValue { field: &'static str },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooSmall { needed } => {
                write!(formatter, "output buffer too small, need {needed} bytes")
            }
            Self::ValueTooLarge { field } => write!(formatter, "value too large for {field}"),
            Self::InvalidValue { field } => write!(formatter, "invalid value for {field}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EncodeError {}
