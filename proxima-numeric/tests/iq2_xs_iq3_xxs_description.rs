#![allow(clippy::expect_used)]

//! `NumericFormat::{Iq2Xs, Iq3Xxs}` describe llama.cpp's `block_iq2_xs` and `block_iq3_xxs` as
//! data. Every expectation below is a struct member of `ggml-common.h`
//! (`/Users/brianbruggeman/repos/others/llama-flint`, tree `e95426f16`): `block_iq2_xs` :387-392
//! (`static_assert` :393), `block_iq3_xxs` :405-410 (:411), indexed the way `dequantize_row_iq2_xs`
//! (`ggml-quants.c:2532-2557`) and `dequantize_row_iq3_xxs` (:2591-2620) index them.
//!
//! Nothing here decodes: this file checks that the description says where each member lives and
//! that the generic walker refuses to read it.

use proxima_numeric::tables::{IQ2XS_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS};
use proxima_numeric::{
    BlockLayout, LevelFormat, LevelGeometry, LevelTable, NumericError, NumericFormat, Plane,
    decode_blocks, encode_blocks,
};

const QK_K: usize = 256;

const TWO: [NumericFormat; 2] = [NumericFormat::Iq2Xs, NumericFormat::Iq3Xxs];

/// One member of a ggml block struct: byte offset and byte length.
struct Member {
    name: &'static str,
    offset: usize,
    bytes: usize,
}

const fn member(name: &'static str, offset: usize, bytes: usize) -> Member {
    Member {
        name,
        offset,
        bytes,
    }
}

/// `block_iq2_xs`: d, qs[QK_K/8] of `uint16_t`, scales[QK_K/32] (`ggml-common.h:387-392`).
const IQ2_XS_MEMBERS: [Member; 3] = [
    member("d", 0, 2),
    member("qs", 2, QK_K / 8 * 2),
    member("scales", 2 + QK_K / 8 * 2, QK_K / 32),
];

/// `block_iq3_xxs`: d, qs[3*QK_K/8] (`ggml-common.h:405-410`).
const IQ3_XXS_MEMBERS: [Member; 2] = [member("d", 0, 2), member("qs", 2, 3 * QK_K / 8)];

/// The bits a described plane claims, and the struct member they must fall inside.
struct Claim {
    plane: &'static str,
    member: &'static str,
    bits: Vec<usize>,
}

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
        LevelTable::SignedGrid { sign, .. } | LevelTable::SignedGrid64 { sign, .. } => sign,
        other => panic!("{other:?} carries no sign plane"),
    }
}

/// The bit addresses `plane` reads over the element indexes `0, step, 2*step, ...` below `limit`.
fn plane_bits(plane: &Plane, limit: usize, step: usize) -> Vec<usize> {
    (0..limit)
        .step_by(step)
        .flat_map(|index| {
            let start = plane.address(index);
            start..start + usize::from(plane.bits)
        })
        .collect()
}

fn claim(
    plane: &'static str,
    member: &'static str,
    plane_value: &Plane,
    limit: usize,
    step: usize,
) -> Claim {
    Claim {
        plane,
        member,
        bits: plane_bits(plane_value, limit, step),
    }
}

fn super_scale_claim(layout: &BlockLayout) -> Claim {
    let start = usize::from(layout.scale.byte_offset) * 8;
    let width = usize::from(layout.scale.layout.width());
    Claim {
        plane: "super scale",
        member: "d",
        bits: (start..start + width).collect(),
    }
}

/// Every plane the format's description names, with the ggml member it must live in.
fn claims_of(format: NumericFormat) -> Vec<Claim> {
    let layout = layout_of(format);
    let sign = sign_plane_of(format);
    let sub = layout.sub.as_ref().expect("these formats have sub-block scales");
    let mut claims = Vec::new();
    match format {
        NumericFormat::Iq2Xs => {
            let high = layout.element.high.expect("iq2_xs has a ninth index bit");
            claims.push(claim("grid index low", "qs", &layout.element.low, QK_K, 8));
            claims.push(claim("grid index high", "qs", &high, QK_K, 8));
            claims.push(claim("sign index", "qs", &sign, QK_K, 8));
            claims.push(claim("sub-block scale", "scales", &sub.scale.low, 16, 1));
        }
        NumericFormat::Iq3Xxs => {
            claims.push(claim("grid index", "qs", &layout.element.low, QK_K, 4));
            claims.push(claim("sign index", "qs", &sign, QK_K, 8));
            claims.push(claim("sub-block scale", "qs", &sub.scale.low, 8, 1));
        }
        other => panic!("{other:?} is not one of the two"),
    }
    claims.push(super_scale_claim(&layout));
    claims
}

fn members_of(format: NumericFormat) -> &'static [Member] {
    match format {
        NumericFormat::Iq2Xs => &IQ2_XS_MEMBERS,
        NumericFormat::Iq3Xxs => &IQ3_XXS_MEMBERS,
        other => panic!("{other:?} is not one of the two"),
    }
}

/// Names the first disagreement between the description and the ggml struct: a plane outside its
/// member, a member bit claimed twice or by nobody, or a wrong block size. `None` when all agree.
fn mismatch(format: NumericFormat, claims: &[Claim]) -> Option<&'static str> {
    let layout = layout_of(format);
    let members = members_of(format);
    let last = members.last().expect("a struct has members");
    if usize::from(layout.block_bytes) != last.offset + last.bytes {
        return Some("block_bytes");
    }
    if usize::from(layout.elements) != QK_K {
        return Some("elements");
    }
    for member in members {
        let low = member.offset * 8;
        let high = (member.offset + member.bytes) * 8;
        let mut seen = vec![0_u8; high - low];
        for owned in claims.iter().filter(|claim| claim.member == member.name) {
            for bit in &owned.bits {
                if *bit < low || *bit >= high {
                    return Some(owned.plane);
                }
                seen[bit - low] += 1;
            }
        }
        if seen.iter().any(|count| *count != 1) {
            return Some(member.name);
        }
    }
    None
}

#[test]
fn iq2_xs_description_claims_every_bit_of_block_iq2_xs_exactly_once() {
    assert_eq!(
        mismatch(NumericFormat::Iq2Xs, &claims_of(NumericFormat::Iq2Xs)),
        None
    );
}

#[test]
fn iq3_xxs_description_claims_every_bit_of_block_iq3_xxs_exactly_once() {
    assert_eq!(
        mismatch(NumericFormat::Iq3Xxs, &claims_of(NumericFormat::Iq3Xxs)),
        None
    );
}

#[test]
fn field_counts_per_member_follow_the_struct_declarations() {
    // (format, member, plane reads claimed) -- entries are plane reads, not bits
    let expected = [
        (NumericFormat::Iq2Xs, "qs", 32 + 32 + 32),
        (NumericFormat::Iq2Xs, "scales", 16),
        (NumericFormat::Iq3Xxs, "qs", 64 + 32 + 8),
    ];
    for (format, name, entries) in expected {
        let found: usize = claims_of(format)
            .iter()
            .filter(|claim| claim.member == name)
            .map(|claim| {
                let width = match claim.plane {
                    "grid index low" | "grid index" => 8,
                    "grid index high" => 1,
                    "sign index" => 7,
                    _ => 4,
                };
                claim.bits.len() / width
            })
            .sum();
        assert_eq!(found, entries, "{format:?} {name}");
    }
}

#[test]
fn iq2_xs_addresses_follow_dequantize_row_iq2_xs() {
    let format = NumericFormat::Iq2Xs;
    let layout = layout_of(format);
    let sign = sign_plane_of(format);
    let sub = layout.sub.expect("iq2_xs has sub-block scales");
    let high = layout.element.high.expect("high plane");

    for element in 0..QK_K {
        // ggml-quants.c:2546-2548: qs[4*ib32 + l] is one u16 per eight elements; grid index
        // qs & 511, sign index qs >> 9
        let word = (2 + 2 * (element / 8)) * 8;
        assert_eq!(layout.element.low.address(element), word, "grid low of {element}");
        assert_eq!(high.address(element), word + 8, "grid ninth bit of {element}");
        assert_eq!(sign.address(element), word + 9, "sign index of {element}");
    }
    for sub_block in 0..16 {
        // ggml-quants.c:2544-2545, 2551: scales[ib32] low nibble then high, db[l/2]
        assert_eq!(
            sub.scale.low.address(sub_block),
            (66 + sub_block / 2) * 8 + 4 * (sub_block % 2)
        );
    }
    assert_eq!((sub.scale.bias, sub.scale_multiplier, sub.count), (-1, 2, 16));
}

#[test]
fn iq3_xxs_addresses_follow_dequantize_row_iq3_xxs() {
    let format = NumericFormat::Iq3Xxs;
    let layout = layout_of(format);
    let sign = sign_plane_of(format);
    let sub = layout.sub.expect("iq3_xxs has sub-block scales");

    for element in 0..QK_K {
        // ggml-quants.c:2606-2607, 2615: qs[2*l] feeds elements j<4 and qs[2*l+1] elements j>=4
        // of each group, qs advancing 8 per ib32, so one grid byte per four elements
        let (ib32, group, half) = (element / 32, element / 8 % 4, element / 4 % 2);
        let grid_byte = 2 + 8 * ib32 + 2 * group + half;
        assert_eq!(layout.element.low.address(element), grid_byte * 8, "grid of {element}");
        // ggml-quants.c:2601, 2605: aux32 at scales_and_signs + 4*ib32, sign index at bit 7*l
        let aux_bit = (2 + QK_K / 4 + 4 * ib32) * 8;
        assert_eq!(sign.address(element), aux_bit + 7 * group, "sign index of {element}");
    }
    for sub_block in 0..8 {
        // ggml-quants.c:2603: aux32 >> 28
        assert_eq!(
            sub.scale.low.address(sub_block),
            (2 + QK_K / 4 + 4 * sub_block) * 8 + 28
        );
    }
    assert_eq!((sub.scale.bias, sub.scale_multiplier, sub.count), (-1, 2, 8));
}

#[test]
fn level_maps_are_the_trained_tables_and_the_stated_constants() {
    let LevelTable::SignedGrid64 {
        grid,
        sign_table,
        unit_shift,
        ..
    } = levels_of(NumericFormat::Iq2Xs).levels
    else {
        panic!("iq2_xs is a signed byte grid");
    };
    // dequantize_row_iq2_xs :2544-2545: * 0.25 on (0.5 + nibble) -> 2^-3 on (2 * nibble + 1)
    assert_eq!((grid, sign_table, unit_shift), (&IQ2XS_GRID[..], Some(&KSIGNS_IQ2XS[..]), 3));

    let LevelTable::SignedGrid {
        grid,
        lanes,
        sign_table,
        unit_shift,
        ..
    } = levels_of(NumericFormat::Iq3Xxs).levels
    else {
        panic!("iq3_xxs is a signed grid");
    };
    // dequantize_row_iq3_xxs :2603: * 0.5 on (0.5 + nibble) -> 2^-2 on (2 * nibble + 1)
    assert_eq!(
        (grid, lanes, sign_table, unit_shift),
        (&IQ3XXS_GRID[..], 4, Some(&KSIGNS_IQ2XS[..]), 2)
    );
}

#[test]
fn grid_tables_are_verbatim_at_their_ends() {
    assert_eq!((IQ2XS_GRID.len(), IQ3XXS_GRID.len()), (512, 256));
    // ggml-common.h:628 and :755, :1018 and :1049
    assert_eq!(IQ2XS_GRID[0], 0x0808_0808_0808_0808);
    assert_eq!(IQ2XS_GRID[511], 0x2b2b_2b2b_2b2b_2b2b);
    assert_eq!(IQ3XXS_GRID[0], 0x0404_0404);
    assert_eq!(IQ3XXS_GRID[255], 0x3e34_1c04);
}

#[test]
fn derived_block_geometry_is_unchanged() {
    for (format, bytes) in [(NumericFormat::Iq2Xs, 74), (NumericFormat::Iq3Xxs, 98)] {
        assert_eq!(format.bytes_per_block(), bytes, "{format:?} bytes");
        assert_eq!(format.elements_per_block(), 256, "{format:?} elements");
        assert_eq!(
            format.packed_len(512),
            Some(usize::from(bytes) * 2),
            "{format:?} packed_len"
        );
    }
}

/// An exhaustive match over `LevelTable`: this function stops compiling if a variant is added,
/// and none of the arms is named after a format.
fn describes_its_codes(table: &LevelTable) -> bool {
    match table {
        LevelTable::Linear { .. }
        | LevelTable::Explicit(_)
        | LevelTable::SignedGrid { .. }
        | LevelTable::SignedGrid64 { .. }
        | LevelTable::DeltaGrid { .. } => true,
    }
}

#[test]
fn no_level_table_variant_is_named_after_a_format() {
    let mut level_formats = 0;
    for format in NumericFormat::ALL {
        if let NumericFormat::Levels(levels) = format {
            level_formats += 1;
            assert!(describes_its_codes(&levels.levels), "{format:?}");
            assert!(
                matches!(levels.geometry, LevelGeometry::Packed(_)),
                "{format:?} carries its packing"
            );
        }
    }
    assert!(level_formats > 0, "no level format in ALL");
}

#[test]
fn control_one_changed_offset_fails_the_member_comparison() {
    for (format, plane, shift) in [
        (NumericFormat::Iq2Xs, "sign index", 8),
        (NumericFormat::Iq2Xs, "sub-block scale", 8),
        (NumericFormat::Iq3Xxs, "sign index", 8),
        (NumericFormat::Iq3Xxs, "sub-block scale", 8),
    ] {
        let mut claims = claims_of(format);
        let target = claims
            .iter_mut()
            .find(|claim| claim.plane == plane)
            .expect("the plane exists");
        for bit in &mut target.bits {
            *bit += shift;
        }
        assert!(mismatch(format, &claims).is_some(), "{format:?} {plane} moved by {shift}");
    }
    for format in TWO {
        assert_eq!(mismatch(format, &claims_of(format)), None, "{format:?} untouched");
    }
}

#[test]
fn the_crate_describes_these_formats_without_decoding_or_encoding_them() {
    let mut values = [0.0_f32; QK_K];
    for format in TWO {
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
fn block_layout_is_none_for_the_two_but_their_packing_is_kept() {
    for format in TWO {
        assert_eq!(format.block_layout(), None, "{format:?}");
        assert!(levels_of(format).packing().is_some(), "{format:?}");
    }
}
