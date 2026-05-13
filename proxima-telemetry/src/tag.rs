use bytes::Bytes;

/// A scalar attribute value; covers ~95% of real-world key/value pairs.
///
/// `Bytes` is Arc-backed: cloning is a refcount bump, no heap copy.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    I64(i64),
    U64(u64),
    F64(f64),
    Bool(bool),
    Str(&'static str),
    Bytes(Bytes),
}

impl core::fmt::Display for ScalarValue {
    /// The bare value — `5`, `control`, `true` — for human/log rendering, NOT the
    /// `Debug` `U64(5)` form. The canonical clean render: formatters and the test
    /// dumper share this instead of re-matching the variants. No alloc (non-UTF-8
    /// bytes fall back to a length placeholder rather than a lossy copy).
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::I64(raw) => write!(f, "{raw}"),
            Self::U64(raw) => write!(f, "{raw}"),
            Self::F64(raw) => write!(f, "{raw}"),
            Self::Bool(raw) => write!(f, "{raw}"),
            Self::Str(text) => f.write_str(text),
            Self::Bytes(raw) => match core::str::from_utf8(raw) {
                Ok(text) => f.write_str(text),
                Err(_) => write!(f, "<{} bytes>", raw.len()),
            },
        }
    }
}

/// A nested attribute value for structured/array payloads.
///
/// Only paid when actually constructed; the static-borrow variants are zero-alloc.
#[derive(Debug, Clone, PartialEq)]
pub enum NestedValue {
    Scalar(ScalarValue),
    Array(&'static [NestedValue]),
    Kv(&'static [(&'static str, NestedValue)]),
}

/// A single key/value attribute attached to a span, metric, or log record.
///
/// `Scalar` covers the hot path; `Structured` is only paid when used.
/// Keys are always `&'static str` — no runtime allocation on Tag construction.
#[derive(Debug, Clone, PartialEq)]
pub enum Tag {
    Scalar {
        key: &'static str,
        value: ScalarValue,
    },
    Structured {
        key: &'static str,
        value: NestedValue,
    },
}

/// Receiver side for a stream of `Tag` values.
///
/// A blanket impl exists for `Vec<Tag>` and `&mut S where S: TagSink`.
pub trait TagSink {
    fn push_tag(&mut self, tag: Tag);
}

impl TagSink for alloc::vec::Vec<Tag> {
    fn push_tag(&mut self, tag: Tag) {
        self.push(tag);
    }
}

impl<S: TagSink + ?Sized> TagSink for &mut S {
    fn push_tag(&mut self, tag: Tag) {
        (**self).push_tag(tag);
    }
}

impl From<i64> for ScalarValue {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<u64> for ScalarValue {
    fn from(value: u64) -> Self {
        Self::U64(value)
    }
}

impl From<f64> for ScalarValue {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}

impl From<bool> for ScalarValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<&'static str> for ScalarValue {
    fn from(value: &'static str) -> Self {
        Self::Str(value)
    }
}

impl From<Bytes> for ScalarValue {
    fn from(value: Bytes) -> Self {
        Self::Bytes(value)
    }
}

impl From<i8> for ScalarValue {
    fn from(value: i8) -> Self {
        Self::I64(i64::from(value))
    }
}

impl From<i16> for ScalarValue {
    fn from(value: i16) -> Self {
        Self::I64(i64::from(value))
    }
}

impl From<i32> for ScalarValue {
    fn from(value: i32) -> Self {
        Self::I64(i64::from(value))
    }
}

impl From<isize> for ScalarValue {
    fn from(value: isize) -> Self {
        Self::I64(value as i64)
    }
}

impl From<u8> for ScalarValue {
    fn from(value: u8) -> Self {
        Self::U64(u64::from(value))
    }
}

impl From<u16> for ScalarValue {
    fn from(value: u16) -> Self {
        Self::U64(u64::from(value))
    }
}

impl From<u32> for ScalarValue {
    fn from(value: u32) -> Self {
        Self::U64(u64::from(value))
    }
}

impl From<usize> for ScalarValue {
    fn from(value: usize) -> Self {
        Self::U64(value as u64)
    }
}

impl From<f32> for ScalarValue {
    fn from(value: f32) -> Self {
        Self::F64(f64::from(value))
    }
}

// reference forms, so the log macros accept `field = &x` the way `tracing`'s
// `Value` does for `&T`.
macro_rules! scalar_value_from_ref {
    ($($ty:ty),* $(,)?) => {
        $(
            impl From<&$ty> for ScalarValue {
                fn from(value: &$ty) -> Self {
                    (*value).into()
                }
            }
        )*
    };
}

scalar_value_from_ref!(
    i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64, bool
);

impl<V: Into<ScalarValue>> From<(&'static str, V)> for Tag {
    fn from((key, value): (&'static str, V)) -> Self {
        Self::Scalar {
            key,
            value: value.into(),
        }
    }
}

/// Push tags into a `TagSink`.
///
/// Accepts: empty, `"key" = value` pairs, bare tag expressions (`Into<Tag>`),
/// and splat iterators (`..iter`). Forms can be mixed freely.
///
/// ```rust
/// # use proxima_telemetry::tag::{Tag, ScalarValue, TagSink};
/// let mut sink: Vec<Tag> = Vec::new();
/// proxima_telemetry::tag!(sink, "http.status" = 200u64, "sampled" = true);
/// assert_eq!(sink.len(), 2);
/// ```
#[macro_export]
macro_rules! tag {
    // TT-muncher internal arms — $sink is already &mut S; evaluated exactly once.
    // All arms produce () without trailing semicolons to stay in expression position.

    // base: nothing left
    (@push $sink:expr $(,)?) => { () };

    // splat (terminal)
    (@push $sink:expr, ..$iter:expr $(,)?) => {{
        for __tag in $iter {
            $crate::tag::TagSink::push_tag($sink, __tag);
        }
    }};

    // k=v (terminal)
    (@push $sink:expr, $key:literal = $value:expr $(,)?) => {{
        $crate::tag::TagSink::push_tag(
            $sink,
            $crate::tag::Tag::Scalar {
                key: $key,
                value: $crate::tag::ScalarValue::from($value),
            },
        );
    }};

    // expression (terminal)
    (@push $sink:expr, $expr:expr $(,)?) => {{
        $crate::tag::TagSink::push_tag($sink, ::core::convert::Into::into($expr));
    }};

    // splat then rest
    (@push $sink:expr, ..$iter:expr, $($rest:tt)+) => {{
        for __tag in $iter {
            $crate::tag::TagSink::push_tag($sink, __tag);
        }
        $crate::tag!(@push $sink, $($rest)+)
    }};

    // k=v then rest
    (@push $sink:expr, $key:literal = $value:expr, $($rest:tt)+) => {{
        $crate::tag::TagSink::push_tag(
            $sink,
            $crate::tag::Tag::Scalar {
                key: $key,
                value: $crate::tag::ScalarValue::from($value),
            },
        );
        $crate::tag!(@push $sink, $($rest)+)
    }};

    // expression then rest
    (@push $sink:expr, $expr:expr, $($rest:tt)+) => {{
        $crate::tag::TagSink::push_tag($sink, ::core::convert::Into::into($expr));
        $crate::tag!(@push $sink, $($rest)+)
    }};

    // public entry: evaluate $sink exactly once, then munch
    ($sink:expr $(,)?) => {{ let _ = &mut $sink; }};

    ($sink:expr, $($rest:tt)+) => {{
        let __sink = &mut $sink;
        $crate::tag!(@push __sink, $($rest)+)
    }};
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]

    use alloc::vec;
    use alloc::vec::Vec;

    use bytes::Bytes;
    use rstest::rstest;

    use super::{NestedValue, ScalarValue, Tag};

    // 1. Tag::Scalar with ScalarValue::I64 constructs and fields are readable
    #[test]
    fn scalar_i64_constructs_and_reads() {
        let tag = Tag::Scalar {
            key: "count",
            value: ScalarValue::I64(42),
        };
        let Tag::Scalar { key, value } = &tag else {
            panic!("wrong variant")
        };
        assert_eq!(*key, "count");
        assert_eq!(*value, ScalarValue::I64(42));
    }

    // 2. Tag::Structured with NestedValue::Array constructs with static-borrowed array
    #[test]
    fn structured_array_constructs() {
        static ITEMS: &[NestedValue] = &[
            NestedValue::Scalar(ScalarValue::I64(1)),
            NestedValue::Scalar(ScalarValue::Bool(true)),
        ];
        let tag = Tag::Structured {
            key: "dims",
            value: NestedValue::Array(ITEMS),
        };
        let Tag::Structured { key, value } = &tag else {
            panic!("wrong variant")
        };
        assert_eq!(*key, "dims");
        let NestedValue::Array(items) = value else {
            panic!("wrong nested variant")
        };
        assert_eq!(items.len(), 2);
    }

    // 3. each ScalarValue variant constructs from its native type via From
    #[rstest]
    #[case::i64(ScalarValue::from(7i64), ScalarValue::I64(7))]
    #[case::u64(ScalarValue::from(8u64), ScalarValue::U64(8))]
    #[case::f64(ScalarValue::from(1.5f64), ScalarValue::F64(1.5))]
    #[case::bool_true(ScalarValue::from(true), ScalarValue::Bool(true))]
    #[case::bool_false(ScalarValue::from(false), ScalarValue::Bool(false))]
    #[case::str_static(ScalarValue::from("hello"), ScalarValue::Str("hello"))]
    #[case::bytes(
        ScalarValue::from(Bytes::from_static(b"raw")),
        ScalarValue::Bytes(Bytes::from_static(b"raw"))
    )]
    fn scalar_value_from_native(#[case] got: ScalarValue, #[case] expected: ScalarValue) {
        assert_eq!(got, expected);
    }

    // 4. tag!(sink, "k" = 42i64) pushes one Tag::Scalar with the right key and value
    #[test]
    fn macro_single_kv_pushes_scalar() {
        let mut sink: Vec<Tag> = Vec::new();
        tag!(sink, "latency" = 42i64);
        assert_eq!(sink.len(), 1);
        let Tag::Scalar { key, value } = &sink[0] else {
            panic!("wrong variant")
        };
        assert_eq!(*key, "latency");
        assert_eq!(*value, ScalarValue::I64(42));
    }

    // 5. tag!(sink, k1=v1, k2=v2, k3=v3) pushes 3 tags in order
    #[test]
    fn macro_multi_kv_pushes_in_order() {
        let mut sink: Vec<Tag> = Vec::new();
        tag!(sink, "k1" = 1i64, "k2" = "two", "k3" = true);
        assert_eq!(sink.len(), 3);

        let Tag::Scalar { key: k1, value: v1 } = &sink[0] else {
            panic!()
        };
        assert_eq!(*k1, "k1");
        assert_eq!(*v1, ScalarValue::I64(1));

        let Tag::Scalar { key: k2, value: v2 } = &sink[1] else {
            panic!()
        };
        assert_eq!(*k2, "k2");
        assert_eq!(*v2, ScalarValue::Str("two"));

        let Tag::Scalar { key: k3, value: v3 } = &sink[2] else {
            panic!()
        };
        assert_eq!(*k3, "k3");
        assert_eq!(*v3, ScalarValue::Bool(true));
    }

    // 6. expression form: tag!(sink, Tag::Scalar { ... }) pushes the expression directly
    #[test]
    fn macro_expression_form_pushes_directly() {
        let mut sink: Vec<Tag> = Vec::new();
        tag!(
            sink,
            Tag::Scalar {
                key: "x",
                value: ScalarValue::F64(1.0)
            }
        );
        assert_eq!(sink.len(), 1);
        let Tag::Scalar { key, value } = &sink[0] else {
            panic!()
        };
        assert_eq!(*key, "x");
        assert_eq!(*value, ScalarValue::F64(1.0));
    }

    // 7. splat form pushes all tags from an iterator
    #[test]
    fn macro_splat_form_pushes_all() {
        let extra = vec![
            Tag::Scalar {
                key: "a",
                value: ScalarValue::I64(1),
            },
            Tag::Scalar {
                key: "b",
                value: ScalarValue::Bool(false),
            },
        ];
        let mut sink: Vec<Tag> = Vec::new();
        tag!(sink, ..extra);
        assert_eq!(sink.len(), 2);
        assert_eq!(
            sink[0],
            Tag::Scalar {
                key: "a",
                value: ScalarValue::I64(1)
            }
        );
        assert_eq!(
            sink[1],
            Tag::Scalar {
                key: "b",
                value: ScalarValue::Bool(false)
            }
        );
    }

    // 8. mixed: kvs + expression + splat in one call
    #[test]
    fn macro_mixed_forms_in_one_call() {
        let extra = vec![Tag::Scalar {
            key: "z",
            value: ScalarValue::U64(99),
        }];
        let mut sink: Vec<Tag> = Vec::new();
        tag!(
            sink,
            "alpha" = 1i64,
            Tag::Scalar {
                key: "beta",
                value: ScalarValue::Str("v")
            },
            ..extra,
            "gamma" = true,
        );
        assert_eq!(sink.len(), 4);
        assert_eq!(
            sink[0],
            Tag::Scalar {
                key: "alpha",
                value: ScalarValue::I64(1)
            }
        );
        assert_eq!(
            sink[1],
            Tag::Scalar {
                key: "beta",
                value: ScalarValue::Str("v")
            }
        );
        assert_eq!(
            sink[2],
            Tag::Scalar {
                key: "z",
                value: ScalarValue::U64(99)
            }
        );
        assert_eq!(
            sink[3],
            Tag::Scalar {
                key: "gamma",
                value: ScalarValue::Bool(true)
            }
        );
    }

    // 9. size assertion — regression guard for Tag's memory layout.
    // on 64-bit: Bytes=32, ScalarValue=40 (Bytes+disc+pad), NestedValue=40 (niche-opt),
    // Tag = key(16) + value(40) + disc/pad(8) = 64B total.
    #[test]
    fn tag_size_is_known() {
        // baseline: 64B on 64-bit (key=16, value=40, disc+pad=8).
        // ScalarValue=40 (Bytes(32)+disc+pad), NestedValue=40 (niche-optimized from ScalarValue).
        // if this assertion fails, Tag's layout changed — update the comment and the constant.
        assert_eq!(core::mem::size_of::<Tag>(), 64);
        assert_eq!(core::mem::size_of::<ScalarValue>(), 40);
        assert_eq!(core::mem::size_of::<NestedValue>(), 40);
    }
}
