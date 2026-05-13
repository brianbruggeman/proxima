//! Owned, `'static` RESP value — the typed protocol-out that rides a
//! `proxima_primitives::pipe::body::Carry` across the async boundary.
//!
//! [`Frame`](super::Frame) borrows from the parse buffer (zero-copy hot path);
//! a `Carry` requires `Send + Sync + 'static`, so the driver lowers the
//! borrowed frame to this owned mirror with [`RespValue::from_frame`] exactly
//! once, at the ownership boundary. Protocol-out is NOT pinned to protocol-in:
//! a `GET` (a bulk-string request) can answer with any `RespValue` the server
//! returns.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::Frame;

/// An owned RESP value. Aggregates own their children; strings own their bytes.
/// `BulkString` stays `Vec<u8>` (binary safe); the text-typed variants
/// (`SimpleString`, `Error`, `BigNumber`, verbatim text) are lossy-UTF-8
/// `String` because their wire contract is text.
#[derive(Debug, Clone, PartialEq)]
pub enum RespValue {
    /// `_`, `$-1`, or `*-1` — every nil shape folds here.
    Null,
    SimpleString(String),
    Error(String),
    Integer(i64),
    BulkString(Vec<u8>),
    Array(Vec<RespValue>),
    Double(f64),
    Boolean(bool),
    BigNumber(String),
    VerbatimString {
        format: String,
        text: String,
    },
    BlobError(String),
    Map(Vec<(RespValue, RespValue)>),
    Set(Vec<RespValue>),
    Push(Vec<RespValue>),
}

impl RespValue {
    /// Own a borrowed [`Frame`]. Recurses through aggregates; the three nil
    /// shapes (`Null`, `NullBlob`, `NullArray`) collapse to [`RespValue::Null`],
    /// and an `Attribute` (out-of-band metadata) is surfaced as a [`Map`] so the
    /// caller can still read it.
    #[must_use]
    pub fn from_frame(frame: &Frame<'_>) -> Self {
        match frame {
            Frame::SimpleString(bytes) => Self::SimpleString(lossy(bytes)),
            Frame::Error(bytes) => Self::Error(lossy(bytes)),
            Frame::Integer(value) => Self::Integer(*value),
            Frame::BlobString(bytes) => Self::BulkString(bytes.to_vec()),
            Frame::NullBlob | Frame::NullArray | Frame::Null => Self::Null,
            Frame::Array(elements) => Self::Array(own_each(elements)),
            Frame::Double(value) => Self::Double(*value),
            Frame::Boolean(value) => Self::Boolean(*value),
            Frame::BigNumber(digits) => Self::BigNumber(lossy(digits)),
            Frame::VerbatimString { format, text } => Self::VerbatimString {
                format: lossy(format),
                text: lossy(text),
            },
            Frame::BlobError(bytes) => Self::BlobError(lossy(bytes)),
            Frame::Map(pairs) => Self::Map(own_pairs(pairs)),
            Frame::Attribute(pairs) => Self::Map(own_pairs(pairs)),
            Frame::Set(elements) => Self::Set(own_each(elements)),
            Frame::Push(elements) => Self::Push(own_each(elements)),
        }
    }

    /// True for any nil shape.
    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// The error message if this is a `-`/`!` error reply.
    #[must_use]
    pub fn as_error(&self) -> Option<&str> {
        match self {
            Self::Error(message) | Self::BlobError(message) => Some(message),
            _ => None,
        }
    }

    /// Borrow the payload bytes of a bulk string or the text of a
    /// simple/verbatim string — the common "read the value" path.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::BulkString(bytes) => Some(bytes),
            Self::SimpleString(text) | Self::BigNumber(text) => Some(text.as_bytes()),
            Self::VerbatimString { text, .. } => Some(text.as_bytes()),
            _ => None,
        }
    }

    /// UTF-8 view of [`as_bytes`](Self::as_bytes).
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::SimpleString(text) | Self::BigNumber(text) => Some(text),
            Self::VerbatimString { text, .. } => Some(text),
            Self::BulkString(bytes) => core::str::from_utf8(bytes).ok(),
            _ => None,
        }
    }

    /// The integer of an `:`-reply.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => Some(*value),
            _ => None,
        }
    }

    /// The elements of an array / set / push reply.
    #[must_use]
    pub fn as_array(&self) -> Option<&[RespValue]> {
        match self {
            Self::Array(items) | Self::Set(items) | Self::Push(items) => Some(items),
            _ => None,
        }
    }

    /// Encode this owned value back to RESP wire bytes — the inverse of
    /// [`from_frame`](Self::from_frame), used when a stored value is replayed
    /// onto the wire (the pub/sub stream re-emits each pushed value).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode_into(&mut out);
        out
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        let frame = self.as_frame();
        super::encode_into(&frame, out);
    }

    /// Borrow this value as a [`Frame`] for encoding. The aggregate variants
    /// recurse, so the returned frame owns nested `Vec`s of borrowed leaves.
    fn as_frame(&self) -> Frame<'_> {
        match self {
            Self::Null => Frame::Null,
            Self::SimpleString(text) => Frame::SimpleString(text.as_bytes()),
            Self::Error(text) => Frame::Error(text.as_bytes()),
            Self::Integer(value) => Frame::Integer(*value),
            Self::BulkString(bytes) => Frame::BlobString(bytes),
            Self::Array(items) => Frame::Array(items.iter().map(Self::as_frame).collect()),
            Self::Double(value) => Frame::Double(*value),
            Self::Boolean(value) => Frame::Boolean(*value),
            Self::BigNumber(text) => Frame::BigNumber(text.as_bytes()),
            Self::VerbatimString { format, text } => Frame::VerbatimString {
                format: format.as_bytes(),
                text: text.as_bytes(),
            },
            Self::BlobError(text) => Frame::BlobError(text.as_bytes()),
            Self::Map(pairs) => Frame::Map(
                pairs
                    .iter()
                    .map(|(key, value)| (key.as_frame(), value.as_frame()))
                    .collect(),
            ),
            Self::Set(items) => Frame::Set(items.iter().map(Self::as_frame).collect()),
            Self::Push(items) => Frame::Push(items.iter().map(Self::as_frame).collect()),
        }
    }
}

fn own_each(frames: &[Frame<'_>]) -> Vec<RespValue> {
    frames.iter().map(RespValue::from_frame).collect()
}

fn own_pairs(pairs: &[(Frame<'_>, Frame<'_>)]) -> Vec<(RespValue, RespValue)> {
    pairs
        .iter()
        .map(|(key, value)| (RespValue::from_frame(key), RespValue::from_frame(value)))
        .collect()
}

fn lossy(bytes: &[u8]) -> String {
    match core::str::from_utf8(bytes) {
        Ok(text) => text.to_string(),
        Err(_) => String::from_utf8_lossy(bytes).into_owned(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use super::super::parse;
    use alloc::vec;

    #[test]
    fn owns_a_bulk_string_reply() {
        let (frame, _) = parse(b"$5\r\nhello\r\n").unwrap();
        assert_eq!(
            RespValue::from_frame(&frame),
            RespValue::BulkString(b"hello".to_vec())
        );
    }

    #[test]
    fn folds_every_nil_shape_to_null() {
        for wire in [&b"_\r\n"[..], &b"$-1\r\n"[..], &b"*-1\r\n"[..]] {
            let (frame, _) = parse(wire).unwrap();
            assert!(
                RespValue::from_frame(&frame).is_null(),
                "wire {wire:?} -> null"
            );
        }
    }

    #[test]
    fn reads_value_accessors() {
        assert_eq!(RespValue::Integer(42).as_i64(), Some(42));
        assert_eq!(RespValue::BulkString(b"abc".to_vec()).as_str(), Some("abc"));
        assert_eq!(
            RespValue::SimpleString("OK".into()).as_bytes(),
            Some(&b"OK"[..])
        );
        assert_eq!(
            RespValue::Error("ERR boom".into()).as_error(),
            Some("ERR boom")
        );
    }

    #[test]
    fn owns_a_nested_map_reply() {
        let (frame, _) = parse(b"%1\r\n$6\r\nserver\r\n$5\r\nredis\r\n").unwrap();
        assert_eq!(
            RespValue::from_frame(&frame),
            RespValue::Map(vec![(
                RespValue::BulkString(b"server".to_vec()),
                RespValue::BulkString(b"redis".to_vec()),
            )])
        );
    }

    #[test]
    fn re_encodes_to_the_same_wire_bytes() {
        let wire = b"*2\r\n$3\r\nfoo\r\n:7\r\n";
        let (frame, _) = parse(wire).unwrap();
        let owned = RespValue::from_frame(&frame);
        assert_eq!(owned.encode(), wire);
    }
}
