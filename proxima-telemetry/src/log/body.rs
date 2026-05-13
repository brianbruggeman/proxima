use bytes::Bytes;

use crate::tag::NestedValue;

/// The payload of a log record.
///
/// Variants are ordered cheapest to most expensive — Empty and Text are zero-alloc.
/// `Owned` is Arc-backed (no copy on clone); `Structured` reuses C4's NestedValue.
#[derive(Debug, Clone, PartialEq)]
pub enum LogBody {
    Empty,
    Text(&'static str),
    Owned(Bytes),
    Structured(NestedValue),
}
