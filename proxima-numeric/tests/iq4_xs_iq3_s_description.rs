#![allow(clippy::expect_used)]

//! `NumericFormat::Iq4Xs` and `NumericFormat::Iq3S` describe llama.cpp's `block_iq4_xs` and
//! `block_iq3_s` as data. Every expectation below is a struct member of `ggml-common.h`
//! (`/Users/brianbruggeman/repos/others/llama.cpp`, tree `f1ea20621`): `block_iq3_s` is
//! :415-421 with its `static_assert` at :422, `block_iq4_xs` is :454-459 with :460.
//!
//! Nothing here decodes: decoding is proxima-gguf's job, checked against llama.cpp's own output by
//! its `iq4_xs_iq3_s_llama_cpp_oracle` test. This file checks that the description says where
//! each member lives.

use proxima_numeric::tables::{IQ3S_GRID, KVALUES_IQ4_NL};
use proxima_numeric::{
    BlockLayout, LevelFormat, LevelTable, NumericError, NumericFormat, Plane, decode_blocks,
    encode_blocks,
};

const QK_K: usize = 256;

/// One member of a ggml block struct: byte offset, byte length, and the bit-plane indexes that
/// address it (`step` element indexes per plane entry).
struct Member {
    name: &'static str,
    offset: usize,
    bytes: usize,
}

/// `block_iq4_xs`: d, scales_h, scales_l[QK_K/64], qs[QK_K/2] (`ggml-common.h:454-459`).
const IQ4_XS_MEMBERS: [Member; 4] = [
    Member {
        name: "d",
        offset: 0,
        bytes: 2,
    },
    Member {
        name: "scales_h",
        offset: 2,
        bytes: 2,
    },
    Member {
        name: "scales_l",
        offset: 4,
        bytes: QK_K / 64,
    },
    Member {
        name: "qs",
        offset: 4 + QK_K / 64,
        bytes: QK_K / 2,
    },
];

/// `block_iq3_s`: d, qs[QK_K/4], qh[QK_K/32], signs[QK_K/8], scales[QK_K/64]
/// (`ggml-common.h:415-421`).
const IQ3_S_MEMBERS: [Member; 5] = [
    Member {
        name: "d",
        offset: 0,
        bytes: 2,
    },
    Member {
        name: "qs",
        offset: 2,
        bytes: QK_K / 4,
    },
    Member {
        name: "qh",
        offset: 2 + QK_K / 4,
        bytes: QK_K / 32,
    },
    Member {
        name: "signs",
        offset: 2 + QK_K / 4 + QK_K / 32,
        bytes: QK_K / 8,
    },
    Member {
        name: "scales",
        offset: 2 + QK_K / 4 + QK_K / 32 + QK_K / 8,
        bytes: QK_K / 64,
    },
];

fn levels_of(format: NumericFormat) -> LevelFormat {
    match format {
        NumericFormat::Levels(levels) => levels,
        other => panic!("{other:?} is not a level format"),
    }
}

fn layout_of(format: NumericFormat) -> BlockLayout {
    levels_of(format)
        .packing()
        .expect("a described format carries its exact packing")
}

fn sign_plane_of(format: NumericFormat) -> Plane {
    match levels_of(format).levels {
        LevelTable::SignedGrid { sign, .. } => sign,
        other => panic!("{other:?} carries no sign plane"),
    }
}

/// The bytes `plane` touches over the element indexes `0, step, 2*step, ... (count-1)*step`, as
/// `(first byte, byte length)`, or `None` when the entries overlap or leave a gap.
fn byte_extent(plane: &Plane, count: usize, step: usize) -> Option<(usize, usize)> {
    let mut covered = vec![false; 1 << 14];
    let mut first = usize::MAX;
    let mut last = 0;
    for index in (0..count).map(|entry| entry * step) {
        let start = plane.address(index);
        for slot in &mut covered[start..start + usize::from(plane.bits)] {
            if *slot {
                return None;
            }
            *slot = true;
        }
        first = first.min(start);
        last = last.max(start + usize::from(plane.bits));
    }
    if (last - first) % 8 != 0 || first % 8 != 0 || last - first != count * usize::from(plane.bits)
    {
        return None;
    }
    Some((first / 8, (last - first) / 8))
}

fn extent_is(member: &Member, found: Option<(usize, usize)>) -> bool {
    found == Some((member.offset, member.bytes))
}

/// Names the first described field that disagrees with the ggml struct, `None` when all agree.
fn iq4_xs_mismatch(layout: &BlockLayout) -> Option<&'static str> {
    let sub = layout.sub.as_ref()?;
    let [d, scales_h, scales_l, qs] = &IQ4_XS_MEMBERS;
    let checks = [
        (
            d.name,
            usize::from(layout.scale.byte_offset) == d.offset
                && usize::from(layout.scale.layout.width()) == d.bytes * 8,
        ),
        (
            scales_h.name,
            sub.scale.high.as_ref().is_some_and(|plane| {
                plane.bits == 2 && extent_is(scales_h, byte_extent(plane, 8, 1))
            }),
        ),
        (
            scales_l.name,
            sub.scale.low.bits == 4 && extent_is(scales_l, byte_extent(&sub.scale.low, 8, 1)),
        ),
        (
            qs.name,
            layout.element.low.bits == 4
                && extent_is(qs, byte_extent(&layout.element.low, QK_K, 1)),
        ),
        (
            "block_bytes",
            usize::from(layout.block_bytes) == qs.offset + qs.bytes,
        ),
        ("elements", usize::from(layout.elements) == QK_K),
        ("sub-block count", usize::from(sub.count) == QK_K / 32),
        (
            "sub-scale bias",
            sub.scale.bias == 32 && sub.scale_multiplier == 1,
        ),
    ];
    checks
        .iter()
        .find(|(_, holds)| !holds)
        .map(|(name, _)| *name)
}

/// As [`iq4_xs_mismatch`], for `block_iq3_s`; `sign` is the format's sign plane.
fn iq3_s_mismatch(layout: &BlockLayout, sign: &Plane) -> Option<&'static str> {
    let sub = layout.sub.as_ref()?;
    let [d, qs, qh, signs, scales] = &IQ3_S_MEMBERS;
    let high = layout.element.high.as_ref()?;
    let checks = [
        (
            d.name,
            usize::from(layout.scale.byte_offset) == d.offset
                && usize::from(layout.scale.layout.width()) == d.bytes * 8,
        ),
        (
            qs.name,
            layout.element.low.bits == 8
                && extent_is(qs, byte_extent(&layout.element.low, QK_K / 4, 4)),
        ),
        (
            qh.name,
            high.bits == 1 && extent_is(qh, byte_extent(high, QK_K / 4, 4)),
        ),
        (
            signs.name,
            sign.bits == 1 && extent_is(signs, byte_extent(sign, QK_K, 1)),
        ),
        (
            scales.name,
            sub.scale.low.bits == 4
                && sub.scale.high.is_none()
                && extent_is(scales, byte_extent(&sub.scale.low, 8, 1)),
        ),
        (
            "block_bytes",
            usize::from(layout.block_bytes) == scales.offset + scales.bytes,
        ),
        ("elements", usize::from(layout.elements) == QK_K),
        ("sub-block count", usize::from(sub.count) == QK_K / 32),
        (
            "sub-scale 2n+1",
            sub.scale.bias == -1 && sub.scale_multiplier == 2,
        ),
    ];
    checks
        .iter()
        .find(|(_, holds)| !holds)
        .map(|(name, _)| *name)
}

#[test]
fn iq4_xs_description_matches_every_member_of_block_iq4_xs() {
    let layout = layout_of(NumericFormat::Iq4Xs);

    assert_eq!(iq4_xs_mismatch(&layout), None);
}

#[test]
fn iq3_s_description_matches_every_member_of_block_iq3_s() {
    let format = NumericFormat::Iq3S;

    assert_eq!(
        iq3_s_mismatch(&layout_of(format), &sign_plane_of(format)),
        None
    );
}

#[test]
fn iq4_xs_addresses_follow_dequantize_row_iq4_xs() {
    let layout = layout_of(NumericFormat::Iq4Xs);
    let sub = layout.sub.expect("iq4_xs has sub-block scales");

    for sub_block in 0..8 {
        // ggml-quants.c:2753: scales_l[ib/2] >> 4*(ib%2), scales_h >> 2*ib
        assert_eq!(
            sub.scale.low.address(sub_block),
            (4 + sub_block / 2) * 8 + 4 * (sub_block % 2)
        );
        assert_eq!(
            sub.scale.high.expect("high plane").address(sub_block),
            16 + 2 * sub_block
        );
    }
    for element in 0..QK_K {
        // ggml-quants.c:2756-2759: first 16 of a sub-block in the low nibbles, last 16 in the high
        let expected = (8 + element / 32 * 16 + element % 16) * 8 + 4 * (element % 32 / 16);
        assert_eq!(
            layout.element.low.address(element),
            expected,
            "element {element}"
        );
    }
}

#[test]
fn iq3_s_addresses_follow_dequantize_row_iq3_s() {
    let format = NumericFormat::Iq3S;
    let layout = layout_of(format);
    let sign = sign_plane_of(format);
    let sub = layout.sub.expect("iq3_s has sub-block scales");

    for sub_block in 0..8 {
        // ggml-quants.c:2619-2620: scales[ib/2] low nibble first
        assert_eq!(
            sub.scale.low.address(sub_block),
            (106 + sub_block / 2) * 8 + 4 * (sub_block % 2)
        );
    }
    for element in 0..QK_K {
        // ggml-quants.c:2622-2634: one qs byte per four elements, its ninth bit is bit i%8 of
        // qh[i/8], one sign bit per element in signs[]
        let index = element / 4;
        assert_eq!(
            layout.element.low.address(element),
            (2 + index) * 8,
            "qs of {element}"
        );
        assert_eq!(
            layout.element.high.expect("high plane").address(element),
            (66 + index / 8) * 8 + index % 8,
            "qh of {element}"
        );
        assert_eq!(sign.address(element), 74 * 8 + element, "sign of {element}");
    }
}

#[test]
fn level_maps_are_the_trained_tables() {
    assert_eq!(
        levels_of(NumericFormat::Iq4Xs).levels,
        LevelTable::Explicit(&KVALUES_IQ4_NL)
    );
    let LevelTable::SignedGrid { grid, lanes, .. } = levels_of(NumericFormat::Iq3S).levels else {
        panic!("iq3_s is a signed grid");
    };
    assert_eq!((grid, lanes), (&IQ3S_GRID[..], 4));
}

#[test]
fn derived_block_geometry_is_unchanged() {
    for (format, bytes) in [(NumericFormat::Iq4Xs, 136), (NumericFormat::Iq3S, 110)] {
        assert_eq!(format.bytes_per_block(), bytes, "{format:?} bytes");
        assert_eq!(format.elements_per_block(), 256, "{format:?} elements");
        assert_eq!(
            format.packed_len(512),
            Some(usize::from(bytes) * 2),
            "{format:?} packed_len"
        );
    }
}

#[test]
fn control_one_changed_offset_fails_the_member_comparison() {
    let mut layout = layout_of(NumericFormat::Iq4Xs);
    let sub = layout.sub.as_mut().expect("iq4_xs has sub-block scales");
    sub.scale.low.base_bits += 8;
    assert_eq!(iq4_xs_mismatch(&layout), Some("scales_l"));

    let format = NumericFormat::Iq3S;
    let mut layout = layout_of(format);
    layout.element.high.as_mut().expect("high plane").base_bits += 8;
    assert_eq!(iq3_s_mismatch(&layout, &sign_plane_of(format)), Some("qh"));

    let mut sign = sign_plane_of(format);
    sign.base_bits += 8;
    assert_eq!(iq3_s_mismatch(&layout_of(format), &sign), Some("signs"));
}

#[test]
fn the_crate_describes_these_formats_without_decoding_or_encoding_them() {
    let mut values = [0.0_f32; QK_K];
    for format in [NumericFormat::Iq4Xs, NumericFormat::Iq3S] {
        let mut bytes = vec![0_u8; usize::from(format.bytes_per_block())];
        assert!(matches!(
            decode_blocks(&format, &bytes, &mut values),
            Err(NumericError::NoDecoder { .. })
        ));
        assert!(matches!(
            encode_blocks(&format, &values, &mut bytes),
            Err(NumericError::UnsupportedEncode { .. })
        ));
    }
}

#[test]
fn block_layout_is_some_only_when_the_layout_alone_determines_the_values() {
    let mut reporting = 0;
    for format in NumericFormat::ALL {
        let Some(layout) = format.block_layout() else {
            continue;
        };
        reporting += 1;
        if let NumericFormat::Levels(levels) = format {
            assert_eq!(
                levels.levels,
                LevelFormat::ladder(&layout),
                "{format:?} reports a layout whose own ladder is not its level table"
            );
        }
    }
    assert!(reporting > 0, "no format reported a block layout");
}

#[test]
fn control_formats_with_a_trained_level_table_report_no_block_layout() {
    for format in [
        NumericFormat::Iq4Nl,
        NumericFormat::Iq4Xs,
        NumericFormat::Iq3S,
    ] {
        assert_eq!(format.block_layout(), None, "{format:?}");
        assert!(
            levels_of(format).packing().is_some(),
            "{format:?} keeps its packing"
        );
    }
    assert_eq!(NumericFormat::Q4_0.block_layout(), Some(BlockLayout::Q4_0));
    assert_eq!(NumericFormat::Tq20.block_layout(), Some(BlockLayout::TQ2_0));
}
