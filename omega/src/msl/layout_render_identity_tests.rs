//! `crate::render` with the Msl dialect renders the same bytes the pre-render emitters did, for
//! every named layout the wide scalar can hold. The expected text was captured from the base
//! emitters into `omega/tests/fixtures/msl_layout/`, one file per layout and function:
//!
//! - `<layout name>.load.msl`: `layout_load_msl(layout, "probe")`, exactly.
//! - `<layout name>.store.msl`: `layout_store_msl(layout, "probe")`, exactly.
//! - `<layout name>.unpack.msl`: `layout_unpack_msl(layout, "probe")` exactly, or the four bytes
//!   `none` (no newline) when the layout has no flat address.
//! - `<layout name>.nan.msl`: `field_store_nan_branch_msl(&layout)`, exactly; present for layouts
//!   with an exponent field only.
//!
//! `<layout name>` is the `NAMED_LAYOUTS` spelling. A missing fixture fails the test.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use std::path::PathBuf;

use proxima_numeric::float_layout::{
    FloatLayout, NAMED_LAYOUTS, NanPolicy, SignEncoding, Specials,
};

use super::dialect::Msl;
use super::layout_render::{
    field_store_nan_branch_msl, layout_load_msl, layout_store_msl, layout_unpack_msl,
};
use crate::render::{self, HardwareScalar, Refusal};

const NO_FLAT_ADDRESS: &str = "none";

fn wide() -> HardwareScalar {
    HardwareScalar::widest().expect("the hardware table has a wide scalar")
}

fn held_by_the_wide_scalar() -> Vec<(FloatLayout, &'static str)> {
    let word_bits = wide().layout().width();
    NAMED_LAYOUTS
        .iter()
        .filter(|(layout, _)| layout.width() <= word_bits)
        .copied()
        .collect()
}

fn fixture(layout_name: &str, function: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/msl_layout")
        .join(format!("{layout_name}.{function}.msl"));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("fixture {} is missing: {error}", path.display()))
}

fn differing(function: &str, render: impl Fn(FloatLayout) -> Option<String>) -> Vec<&'static str> {
    let layouts = held_by_the_wide_scalar();
    assert!(!layouts.is_empty(), "no named layout was compared");
    layouts
        .into_iter()
        .filter(|(layout, name)| {
            let Some(rendered) = render(*layout) else {
                return false;
            };
            fixture(name, function) != rendered
        })
        .map(|(_, name)| name)
        .collect()
}

#[test]
fn the_wide_scalar_holds_a_named_layout_of_every_class() {
    let layouts = held_by_the_wide_scalar();

    let integers = layouts
        .iter()
        .filter(|(layout, _)| !layout.is_float())
        .count();
    let with_exponent = layouts
        .iter()
        .filter(|(layout, _)| layout.is_float())
        .count();

    assert!(integers > 0, "no integer layout was compared");
    assert!(
        with_exponent > 0,
        "no layout with an exponent field was compared"
    );
}

#[test]
fn load_is_byte_identical_to_the_captured_base_output_for_every_named_layout() {
    let differing = differing("load", |layout| Some(layout_load_msl(layout, "probe")));

    assert_eq!(differing, Vec::<&str>::new());
}

#[test]
fn store_is_byte_identical_to_the_captured_base_output_for_every_named_layout() {
    let differing = differing("store", |layout| Some(layout_store_msl(layout, "probe")));

    assert_eq!(differing, Vec::<&str>::new());
}

#[test]
fn unpack_is_byte_identical_to_the_captured_base_output_for_every_named_layout() {
    let differing = differing("unpack", |layout| {
        Some(layout_unpack_msl(layout, "probe").unwrap_or_else(|| String::from(NO_FLAT_ADDRESS)))
    });

    assert_eq!(differing, Vec::<&str>::new());
}

#[test]
fn the_nan_branch_is_byte_identical_to_the_captured_base_output_for_every_layout_with_an_exponent()
{
    let differing = differing("nan", |layout| {
        layout
            .is_float()
            .then(|| field_store_nan_branch_msl(&layout))
    });

    assert_eq!(differing, Vec::<&str>::new());
}

#[test]
fn every_fixture_file_is_read_by_some_comparison() {
    let mut expected: Vec<String> = Vec::new();
    for (layout, name) in held_by_the_wide_scalar() {
        expected.push(format!("{name}.load.msl"));
        expected.push(format!("{name}.store.msl"));
        expected.push(format!("{name}.unpack.msl"));
        if layout.is_float() {
            expected.push(format!("{name}.nan.msl"));
        }
    }
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/msl_layout");

    let mut present: Vec<String> = std::fs::read_dir(&directory)
        .expect("the fixture directory exists")
        .map(|entry| {
            entry
                .expect("a readable directory entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    present.sort();
    expected.sort();

    assert!(!present.is_empty(), "no fixture was found");
    assert_eq!(present, expected);
}

#[test]
fn a_layout_wider_than_the_wide_scalar_is_refused_by_load_store_and_unpack_with_its_width() {
    let word_bits = wide().layout().width();
    let wider: Vec<(FloatLayout, &str)> = NAMED_LAYOUTS
        .iter()
        .filter(|(layout, _)| layout.width() > word_bits)
        .copied()
        .collect();

    assert!(
        !wider.is_empty(),
        "no named layout is wider than the wide scalar"
    );
    for (layout, name) in wider {
        let refusal = Refusal::CodeWiderThanWord {
            width: layout.width(),
        };
        assert_eq!(
            render::load(&Msl, wide(), layout, "probe"),
            Err(refusal),
            "{name} load"
        );
        assert_eq!(
            render::store(&Msl, wide(), layout, "probe"),
            Err(refusal),
            "{name} store"
        );
        assert_eq!(
            render::unpack(&Msl, wide(), layout, "probe"),
            Err(refusal),
            "{name} unpack"
        );
        assert!(
            layout_load_msl(layout, "probe").starts_with("#error"),
            "{name} msl load"
        );
        assert!(
            layout_store_msl(layout, "probe").starts_with("#error"),
            "{name} msl store"
        );
        assert_eq!(
            layout_unpack_msl(layout, "probe"),
            None,
            "{name} msl unpack"
        );
    }
}

#[test]
fn an_element_wider_than_a_byte_that_straddles_no_two_bytes_is_refused_with_its_width() {
    let seventeen_bits = FloatLayout {
        exponent_bits: 5,
        mantissa_bits: 11,
        specials: Specials::InfinityOnOverflow,
        bias_override: None,
        sign: SignEncoding::Magnitude,
        nan: NanPolicy::IEEE,
    };

    let refused = render::unpack(&Msl, wide(), seventeen_bits, "probe");

    assert_eq!(refused, Err(Refusal::NoFlatAddress { width: 17 }));
    assert_eq!(layout_unpack_msl(seventeen_bits, "probe"), None);
}
