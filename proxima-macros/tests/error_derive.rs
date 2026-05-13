#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! End-to-end tests for `#[derive(Error)]`. Verifies that the derive's
//! emitted code compiles, links, and behaves correctly — not just that
//! the macro produces well-formed tokens (the proxima-macros unit tests
//! cover that lower level).

#![allow(dead_code)]

use core::error::Error as _;
use proxima_macros::Error;

#[derive(Debug, Error)]
enum SimpleError {
    #[error("invalid")]
    Invalid,
    #[error("not found")]
    NotFound,
}

#[test]
fn unit_variants_display_their_literal() {
    assert_eq!(format!("{}", SimpleError::Invalid), "invalid");
    assert_eq!(format!("{}", SimpleError::NotFound), "not found");
}

#[test]
fn simple_error_has_no_source() {
    let invalid = SimpleError::Invalid;
    assert!(invalid.source().is_none());
}

#[derive(Debug, Error)]
enum DecodeError {
    #[error("invalid magic byte: {0}")]
    InvalidMagic(u8),
    #[error("truncated frame at offset {0}")]
    Truncated(usize),
    #[error("expected {0:#x} got {1:#x}")]
    Mismatch(u32, u32),
}

#[test]
fn tuple_variants_interpolate_positional_args() {
    assert_eq!(
        format!("{}", DecodeError::InvalidMagic(0x7e)),
        "invalid magic byte: 126"
    );
    assert_eq!(
        format!("{}", DecodeError::Truncated(42)),
        "truncated frame at offset 42"
    );
    assert_eq!(
        format!("{}", DecodeError::Mismatch(0xdeadbeef, 0xcafebabe)),
        "expected 0xdeadbeef got 0xcafebabe"
    );
}

#[derive(Debug, Error)]
enum NamedFieldError {
    #[error("missing header {name}")]
    MissingHeader { name: &'static str },
}

#[test]
fn named_field_variants_interpolate_by_name() {
    let err = NamedFieldError::MissingHeader { name: "x-trace-id" };
    assert_eq!(format!("{err}"), "missing header x-trace-id");
}

#[derive(Debug)]
struct Inner {
    detail: u32,
}

impl core::fmt::Display for Inner {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "inner(detail={})", self.detail)
    }
}

impl core::error::Error for Inner {}

#[derive(Debug, Error)]
enum WithSource {
    #[error("wrapped failure")]
    Wrapped(#[source] Inner),
}

#[test]
fn source_attribute_exposes_inner_via_error_source() {
    let inner = Inner { detail: 7 };
    let err = WithSource::Wrapped(inner);
    let src = err.source().expect("source should be Some");
    assert_eq!(format!("{src}"), "inner(detail=7)");
}

#[derive(Debug, Error)]
enum WithFrom {
    #[error("from-inner")]
    FromInner(#[from] Inner),
}

#[test]
fn from_attribute_generates_from_impl() {
    let inner = Inner { detail: 42 };
    let err: WithFrom = inner.into();
    assert_eq!(format!("{err}"), "from-inner");
    // call `.source()` via the trait so the `Error` trait stays imported
    let src = core::error::Error::source(&err).expect("from implies source");
    assert_eq!(format!("{src}"), "inner(detail=42)");
}

#[derive(Debug, Error)]
enum Transparent {
    #[error(transparent)]
    Forward(Inner),
}

#[test]
fn transparent_forwards_display_and_source_to_inner() {
    let err = Transparent::Forward(Inner { detail: 99 });
    assert_eq!(format!("{err}"), "inner(detail=99)");
    let src = err.source().expect("transparent surfaces inner");
    assert_eq!(format!("{src}"), "inner(detail=99)");
}

#[derive(Debug, Error)]
enum Mixed {
    #[error("a")]
    A,
    #[error("b with {0}")]
    B(u32),
    #[error("c with {name}")]
    C { name: &'static str },
    #[error("wrapped: {0}")]
    D(#[from] Inner),
    #[error(transparent)]
    E(Inner),
}

#[test]
fn mixed_enum_dispatches_per_variant() {
    assert_eq!(format!("{}", Mixed::A), "a");
    assert_eq!(format!("{}", Mixed::B(7)), "b with 7");
    assert_eq!(format!("{}", Mixed::C { name: "x" }), "c with x");
    let from_inner: Mixed = Inner { detail: 1 }.into();
    assert!(matches!(from_inner, Mixed::D(_)));
    let transparent_err = Mixed::E(Inner { detail: 2 });
    assert_eq!(format!("{transparent_err}"), "inner(detail=2)");
}

#[test]
fn mixed_enum_sources_match_per_variant() {
    assert!(Mixed::A.source().is_none());
    assert!(Mixed::B(0).source().is_none());
    assert!(Mixed::C { name: "x" }.source().is_none());
    assert!(Mixed::D(Inner { detail: 0 }).source().is_some());
    assert!(Mixed::E(Inner { detail: 0 }).source().is_some());
}

#[derive(Debug, Error)]
enum Generic<T: core::fmt::Debug + core::fmt::Display + core::error::Error + 'static> {
    #[error(transparent)]
    Wrapped(T),
}

#[test]
fn generic_enums_compile_and_dispatch() {
    let err: Generic<Inner> = Generic::Wrapped(Inner { detail: 5 });
    assert_eq!(format!("{err}"), "inner(detail=5)");
}

#[test]
fn nested_source_chain_walkable_via_core_error_iter() {
    let level0 = Inner { detail: 1 };
    let level1: WithFrom = level0.into();
    // Walk the chain manually because core::error::Iter is unstable.
    let direct = level1.source().expect("level1 has source");
    assert_eq!(format!("{direct}"), "inner(detail=1)");
}
