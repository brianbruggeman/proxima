use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_numeric::float_layout::{NAMED_LAYOUTS, SignEncoding};

use super::dialect::{WORD_BITS, Wgsl};
use crate::render::{self, HardwareScalar, Refusal};

fn wide_row() -> HardwareScalar {
    HardwareScalar::widest(&Wgsl).expect("the wgsl dialect spells one row")
}

fn module_around(functions: &str) -> String {
    format!(
        "@group(0) @binding(0) var<storage, read_write> payload: array<u32>;\n{functions}\n@compute @workgroup_size(1)\nfn main() {{}}\n"
    )
}

fn naga_verdict(source: &str) -> Result<(), String> {
    let module =
        naga::front::wgsl::parse_str(source).map_err(|error| error.emit_to_string(source))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map(|_| ())
    .map_err(|error| format!("{error:?}"))
}

#[test]
fn exactly_one_hardware_row_has_the_wgsl_word_width() {
    let rows: Vec<HardwareScalar> = HardwareScalar::all(&Wgsl).collect();

    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0].layout().width(), WORD_BITS);
}

#[test]
fn a_layout_wider_than_the_wgsl_word_is_refused_with_its_width() {
    let wide: Vec<_> = NAMED_LAYOUTS
        .iter()
        .filter(|(layout, _)| layout.width() > WORD_BITS)
        .collect();

    assert!(!wide.is_empty(), "no named layout is wider than the word");
    for (layout, name) in wide {
        let refused = render::load(&Wgsl, wide_row(), *layout, "probe");
        assert_eq!(
            refused,
            Err(Refusal::CodeWiderThanWord {
                width: layout.width()
            }),
            "{name}"
        );
    }
}

#[test]
fn int32_load_is_refused_for_want_of_a_64_bit_integer() {
    let int32: Vec<_> = NAMED_LAYOUTS
        .iter()
        .filter(|(layout, _)| {
            layout.is_integer()
                && layout.sign == SignEncoding::TwosComplement
                && layout.width() == WORD_BITS
        })
        .collect();

    assert_eq!(int32.len(), 1, "the 32-bit two's-complement layout");
    let refused = render::load(&Wgsl, wide_row(), int32[0].0, "probe");
    assert!(
        matches!(refused, Err(Refusal::NoIntegerOfWidth { .. })),
        "{refused:?}"
    );
}

#[test]
fn every_wgsl_load_store_and_unpack_validates_in_naga() {
    let mut validated = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for (layout, name) in NAMED_LAYOUTS.iter() {
        let renders = [
            ("load", render::load(&Wgsl, wide_row(), *layout, "probe")),
            ("store", render::store(&Wgsl, wide_row(), *layout, "probe")),
            (
                "unpack",
                render::unpack(&Wgsl, wide_row(), *layout, "probe"),
            ),
        ];
        for (step, rendered) in renders {
            let Ok(text) = rendered else { continue };
            match naga_verdict(&module_around(&text)) {
                Ok(()) => validated += 1,
                Err(reason) => failures.push(format!("{name} {step}: {reason}\n{text}")),
            }
        }
    }

    assert!(validated > 0, "no render was validated");
    assert!(failures.is_empty(), "{}", failures.join("\n---\n"));
}
