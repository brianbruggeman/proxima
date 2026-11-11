#![allow(clippy::expect_used)]

//! `NumericFormat::{Iq1S, Iq1M, Iq2Xxs, Iq2S}` describe llama.cpp's `block_iq1_s`, `block_iq1_m`,
//! `block_iq2_xxs` and `block_iq2_s` as data. Every expectation below is a struct member of
//! `ggml-common.h` (`/Users/brianbruggeman/repos/others/llama-flint`, tree `e95426f16`):
//! `block_iq2_xxs` :381-384 (`static_assert` :385), `block_iq2_s` :396-401 (:402), `block_iq1_s`
//! :425-429 (:430), `block_iq1_m` :433-437 (:438), indexed the way `dequantize_row_iq2_xxs`
//! (`ggml-quants.c:2504-2528`), `dequantize_row_iq2_s` (:2559-2587), `dequantize_row_iq1_s`
//! (:2666-2689) and `dequantize_row_iq1_m` (:2691-2739) index them.
//!
//! Nothing here decodes: this file checks that the description says where each member lives and
//! that the generic walker refuses to read it.

use proxima_numeric::tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS,
};
use proxima_numeric::{
    BlockLayout, LevelFormat, LevelGeometry, LevelTable, NumericError, NumericFormat, Plane,
    decode_blocks, encode_blocks,
};

const QK_K: usize = 256;

const FOUR: [NumericFormat; 4] = [
    NumericFormat::Iq1S,
    NumericFormat::Iq1M,
    NumericFormat::Iq2Xxs,
    NumericFormat::Iq2S,
];

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

/// `block_iq2_xxs`: d, qs[QK_K/8] of `uint16_t` (`ggml-common.h:381-384`).
const IQ2_XXS_MEMBERS: [Member; 2] = [member("d", 0, 2), member("qs", 2, QK_K / 8 * 2)];

/// `block_iq2_s`: d, qs[QK_K/4], qh[QK_K/32], scales[QK_K/32] (`ggml-common.h:396-401`).
const IQ2_S_MEMBERS: [Member; 4] = [
    member("d", 0, 2),
    member("qs", 2, QK_K / 4),
    member("qh", 2 + QK_K / 4, QK_K / 32),
    member("scales", 2 + QK_K / 4 + QK_K / 32, QK_K / 32),
];

/// `block_iq1_s`: d, qs[QK_K/8], qh[QK_K/32] of `uint16_t` (`ggml-common.h:425-429`).
const IQ1_S_MEMBERS: [Member; 3] = [
    member("d", 0, 2),
    member("qs", 2, QK_K / 8),
    member("qh", 2 + QK_K / 8, QK_K / 32 * 2),
];

/// `block_iq1_m`: qs[QK_K/8], qh[QK_K/16], scales[QK_K/32]; no `d` (`ggml-common.h:433-437`).
const IQ1_M_MEMBERS: [Member; 3] = [
    member("qs", 0, QK_K / 8),
    member("qh", QK_K / 8, QK_K / 16),
    member("scales", QK_K / 8 + QK_K / 16, QK_K / 32),
];

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

fn scale_bits(layout: &BlockLayout) -> Vec<usize> {
    match layout.scale.pieces {
        Some(pieces) => {
            plane_bits(&pieces, usize::from(layout.scale.layout.width()) / usize::from(pieces.bits), 1)
        }
        None => {
            let start = usize::from(layout.scale.byte_offset) * 8;
            (start..start + usize::from(layout.scale.layout.width())).collect()
        }
    }
}

fn sub_scale_claim(layout: &BlockLayout, member: &'static str) -> Claim {
    let sub = layout.sub.as_ref().expect("these formats have sub-block scales");
    claim("sub-block scale", member, &sub.scale.low, usize::from(sub.count), 1)
}

/// Every plane the format's description names, with the ggml member it must live in.
fn claims_of(format: NumericFormat) -> Vec<Claim> {
    let layout = layout_of(format);
    let levels = levels_of(format).levels;
    let low = claim("grid index low", "qs", &layout.element.low, QK_K, 8);
    let mut claims = vec![low];
    let high = |member: &'static str| {
        let plane = layout.element.high.expect("these codes have a high plane");
        claim("grid index high", member, &plane, QK_K, 8)
    };
    match (format, levels) {
        (NumericFormat::Iq2Xxs, LevelTable::SignedGrid64 { sign, .. }) => {
            claims.push(sub_scale_claim(&layout, "qs"));
            claims.push(claim("sign index", "qs", &sign, QK_K, 8));
        }
        (NumericFormat::Iq2S, LevelTable::SignedGrid64 { sign, .. }) => {
            claims.push(high("qh"));
            claims.push(sub_scale_claim(&layout, "scales"));
            claims.push(claim("signs", "qs", &sign, QK_K, 1));
        }
        (NumericFormat::Iq1S, LevelTable::DeltaGrid { delta, .. }) => {
            claims.push(high("qh"));
            claims.push(sub_scale_claim(&layout, "qh"));
            claims.push(claim("delta sign", "qh", &delta, QK_K, 32));
        }
        (NumericFormat::Iq1M, LevelTable::DeltaGrid { delta, .. }) => {
            claims[0].member = "qs";
            claims.push(high("qh"));
            claims.push(sub_scale_claim(&layout, "scales"));
            claims.push(claim("delta sign", "qh", &delta, QK_K, 8));
        }
        (other, _) => panic!("{other:?} is not one of the four"),
    }
    claims.push(Claim {
        plane: "super scale",
        member: if format == NumericFormat::Iq1M { "scales" } else { "d" },
        bits: scale_bits(&layout),
    });
    claims
}

fn members_of(format: NumericFormat) -> &'static [Member] {
    match format {
        NumericFormat::Iq2Xxs => &IQ2_XXS_MEMBERS,
        NumericFormat::Iq2S => &IQ2_S_MEMBERS,
        NumericFormat::Iq1S => &IQ1_S_MEMBERS,
        NumericFormat::Iq1M => &IQ1_M_MEMBERS,
        other => panic!("{other:?} is not one of the four"),
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
fn iq2_xxs_description_matches_every_member_of_block_iq2_xxs() {
    assert_eq!(
        mismatch(NumericFormat::Iq2Xxs, &claims_of(NumericFormat::Iq2Xxs)),
        None
    );
}

#[test]
fn iq2_s_description_matches_every_member_of_block_iq2_s() {
    assert_eq!(
        mismatch(NumericFormat::Iq2S, &claims_of(NumericFormat::Iq2S)),
        None
    );
}

#[test]
fn iq1_s_description_matches_every_member_of_block_iq1_s() {
    assert_eq!(
        mismatch(NumericFormat::Iq1S, &claims_of(NumericFormat::Iq1S)),
        None
    );
}

#[test]
fn iq1_m_description_matches_every_member_of_block_iq1_m() {
    assert_eq!(
        mismatch(NumericFormat::Iq1M, &claims_of(NumericFormat::Iq1M)),
        None
    );
}

#[test]
fn field_counts_per_member_follow_the_struct_declarations() {
    // (format, member, claimed entries) -- entries are plane reads, not bits
    let expected = [
        (NumericFormat::Iq2Xxs, "qs", 32 + 32 + 8),
        (NumericFormat::Iq2S, "qs", 32 + 256),
        (NumericFormat::Iq2S, "qh", 32),
        (NumericFormat::Iq2S, "scales", 16),
        (NumericFormat::Iq1S, "qs", 32),
        (NumericFormat::Iq1S, "qh", 32 + 8 + 8),
        (NumericFormat::Iq1M, "qs", 32),
        (NumericFormat::Iq1M, "qh", 32 + 32),
        (NumericFormat::Iq1M, "scales", 16 + 4),
    ];
    for (format, name, entries) in expected {
        let claims = claims_of(format);
        let found: usize = claims
            .iter()
            .filter(|claim| claim.member == name)
            .map(|claim| {
                let width = match claim.plane {
                    "grid index low" => 8,
                    "grid index high" => match format {
                        NumericFormat::Iq2S => 2,
                        _ => 3,
                    },
                    "sign index" => 7,
                    "sub-block scale" => match format {
                        NumericFormat::Iq1S | NumericFormat::Iq1M => 3,
                        _ => 4,
                    },
                    "super scale" => 4,
                    _ => 1,
                };
                claim.bits.len() / width
            })
            .sum();
        assert_eq!(found, entries, "{format:?} {name}");
    }
}

#[test]
fn iq2_xxs_addresses_follow_dequantize_row_iq2_xxs() {
    let format = NumericFormat::Iq2Xxs;
    let layout = layout_of(format);
    let LevelTable::SignedGrid64 { sign, .. } = levels_of(format).levels else {
        panic!("iq2_xxs is a signed byte grid");
    };
    let sub = layout.sub.expect("iq2_xxs has sub-block scales");

    for element in 0..QK_K {
        // ggml-quants.c:2512-2520: record ib32 is 8 bytes at qs + 4*ib32 (u16 units): four grid
        // bytes aux8[l], then aux32[1] with the 7-bit sign index at bit 7*l
        let record = 2 + 8 * (element / 32);
        let lane = element % 32 / 8;
        assert_eq!(layout.element.low.address(element), (record + lane) * 8, "grid of {element}");
        assert_eq!(sign.address(element), (record + 4) * 8 + 7 * lane, "sign of {element}");
    }
    for sub_block in 0..8 {
        // ggml-quants.c:2517: aux32[1] >> 28
        assert_eq!(
            sub.scale.low.address(sub_block),
            (2 + 8 * sub_block + 4) * 8 + 28
        );
    }
    assert_eq!((sub.scale.bias, sub.scale_multiplier), (-1, 2));
}

#[test]
fn iq2_s_addresses_follow_dequantize_row_iq2_s() {
    let format = NumericFormat::Iq2S;
    let layout = layout_of(format);
    let LevelTable::SignedGrid64 { sign, .. } = levels_of(format).levels else {
        panic!("iq2_s is a signed byte grid");
    };
    let sub = layout.sub.expect("iq2_s has sub-block scales");
    let high = layout.element.high.expect("high plane");

    for element in 0..QK_K {
        // ggml-quants.c:2570-2577: qs[l] low byte, qh[ib32] bits 2*l, signs[l] at qs + 32
        let group = element / 8;
        let (ib32, lane) = (group / 4, group % 4);
        assert_eq!(layout.element.low.address(element), (2 + group) * 8, "qs of {element}");
        assert_eq!(high.address(element), (66 + ib32) * 8 + 2 * lane, "qh of {element}");
        assert_eq!(sign.address(element), (34 + group) * 8 + element % 8, "sign of {element}");
    }
    for sub_block in 0..16 {
        // ggml-quants.c:2573-2574: scales[ib32] low nibble then high, per 16 elements
        assert_eq!(
            sub.scale.low.address(sub_block),
            (74 + sub_block / 2) * 8 + 4 * (sub_block % 2)
        );
    }
    assert_eq!((sub.scale.bias, sub.scale_multiplier), (-1, 2));
}

#[test]
fn iq1_s_addresses_follow_dequantize_row_iq1_s() {
    let format = NumericFormat::Iq1S;
    let layout = layout_of(format);
    let LevelTable::DeltaGrid { delta, .. } = levels_of(format).levels else {
        panic!("iq1_s is a delta grid");
    };
    let sub = layout.sub.expect("iq1_s has sub-block scales");
    let high = layout.element.high.expect("high plane");

    for element in 0..QK_K {
        // ggml-quants.c:2675-2685: qs += 4 per ib; qh[ib] bits 3*l (index), 12..14 (scale), 15 (delta)
        let group = element / 8;
        let (ib, lane) = (group / 4, group % 4);
        let word = (34 + 2 * ib) * 8;
        assert_eq!(layout.element.low.address(element), (2 + group) * 8, "qs of {element}");
        assert_eq!(high.address(element), word + 3 * lane, "qh index of {element}");
        assert_eq!(delta.address(element), word + 15, "delta of {element}");
    }
    for sub_block in 0..8 {
        assert_eq!(sub.scale.low.address(sub_block), (34 + 2 * sub_block) * 8 + 12);
    }
    assert_eq!((sub.scale.bias, sub.scale_multiplier), (-1, 2));
}

#[test]
fn iq1_m_addresses_follow_dequantize_row_iq1_m() {
    let format = NumericFormat::Iq1M;
    let layout = layout_of(format);
    let LevelTable::DeltaGrid { delta, .. } = levels_of(format).levels else {
        panic!("iq1_m is a delta grid");
    };
    let sub = layout.sub.expect("iq1_m has sub-block scales");
    let high = layout.element.high.expect("high plane");

    for element in 0..QK_K {
        // ggml-quants.c:2713-2720: qs[l]; qh[l/2] nibble l%2: bits 0..2 index, bit 3 delta
        let group = element / 8;
        let (ib, lane) = (group / 4, group % 4);
        let nibble = (32 + 2 * ib + lane / 2) * 8 + 4 * (lane % 2);
        assert_eq!(layout.element.low.address(element), group * 8, "qs of {element}");
        assert_eq!(high.address(element), nibble, "qh index of {element}");
        assert_eq!(delta.address(element), nibble + 3, "delta of {element}");
    }
    for sub_block in 0..16 {
        // ggml-quants.c:2710-2711: sc[ib/2] >> (6*(ib%2) + 3*h), sub-block = 2*ib + h
        let (ib, half) = (sub_block / 2, sub_block % 2);
        assert_eq!(
            sub.scale.low.address(sub_block),
            (48 + 2 * (ib / 2)) * 8 + 6 * (ib % 2) + 3 * half
        );
    }
    // ggml-quants.c:2703: the f16 super-scale is the top nibble of each of sc[0..4], low nibble first
    let pieces = layout.scale.pieces.expect("iq1_m's super-scale is scattered");
    for piece in 0..4 {
        assert_eq!(pieces.address(piece), (48 + 2 * piece) * 8 + 12);
    }
    assert_eq!((sub.scale.bias, sub.scale_multiplier), (-1, 2));
}

#[test]
fn level_maps_are_the_trained_tables_and_the_stated_constants() {
    let LevelTable::SignedGrid64 {
        grid,
        sign_table,
        unit_shift,
        ..
    } = levels_of(NumericFormat::Iq2Xxs).levels
    else {
        panic!("iq2_xxs is a signed byte grid");
    };
    assert_eq!((grid, sign_table, unit_shift), (&IQ2XXS_GRID[..], Some(&KSIGNS_IQ2XS[..]), 3));
    let LevelTable::SignedGrid64 {
        grid,
        sign_table,
        unit_shift,
        ..
    } = levels_of(NumericFormat::Iq2S).levels
    else {
        panic!("iq2_s is a signed byte grid");
    };
    assert_eq!((grid, sign_table, unit_shift), (&IQ2S_GRID[..], None, 3));
    for format in [NumericFormat::Iq1S, NumericFormat::Iq1M] {
        let LevelTable::DeltaGrid {
            grid, delta_shift, ..
        } = levels_of(format).levels
        else {
            panic!("{format:?} is a delta grid");
        };
        // IQ1S_DELTA = IQ1M_DELTA = 0.125 (ggml-common.h:1132-1133)
        assert_eq!((grid, delta_shift), (&IQ1S_GRID[..], 3), "{format:?}");
    }
    assert_eq!(KMASK_IQ2XS, [1, 2, 4, 8, 16, 32, 64, 128]);
}

#[test]
fn grid_tables_are_verbatim_at_their_ends() {
    assert_eq!((IQ2XXS_GRID.len(), IQ2S_GRID.len(), IQ1S_GRID.len()), (256, 1024, 2048));
    // ggml-common.h:561 and :624, :759 and :1014, :1136 and :1647
    assert_eq!(IQ2XXS_GRID[0], 0x0808_0808_0808_0808);
    assert_eq!(IQ2XXS_GRID[255], 0x2b2b_2b19_0808_1908);
    assert_eq!(IQ2S_GRID[0], 0x0808_0808_0808_0808);
    assert_eq!(IQ2S_GRID[1023], 0x2b2b_2b2b_2b2b_2b2b);
    assert_eq!(IQ1S_GRID[0], 0xffff_ffff_ffff_ffff);
    assert_eq!(IQ1S_GRID[2047], 0x0101_0101_0101_0101);
}

#[test]
fn derived_block_geometry_is_unchanged() {
    for (format, bytes) in [
        (NumericFormat::Iq1S, 50),
        (NumericFormat::Iq1M, 56),
        (NumericFormat::Iq2Xxs, 66),
        (NumericFormat::Iq2S, 82),
    ] {
        assert_eq!(format.bytes_per_block(), bytes, "{format:?} bytes");
        assert_eq!(format.elements_per_block(), 256, "{format:?} elements");
        assert_eq!(
            format.packed_len(512),
            Some(usize::from(bytes) * 2),
            "{format:?} packed_len"
        );
    }
}

/// An exhaustive match over `LevelTable`: this function stops compiling if a variant is added or
/// the removed `External` one comes back without being named here.
fn describes_its_codes(table: &LevelTable) -> bool {
    match table {
        LevelTable::Linear { .. }
        | LevelTable::Explicit(_)
        | LevelTable::Iq2XsGrid
        | LevelTable::Iq3XxsGrid
        | LevelTable::SignedGrid { .. }
        | LevelTable::SignedGrid64 { .. }
        | LevelTable::DeltaGrid { .. } => true,
    }
}

#[test]
fn no_format_in_all_is_external() {
    let mut level_formats = 0;
    let mut codebook_only = 0;
    for format in NumericFormat::ALL {
        if let NumericFormat::Levels(levels) = format {
            level_formats += 1;
            assert!(describes_its_codes(&levels.levels), "{format:?}");
            if matches!(levels.geometry, LevelGeometry::Codebook { .. }) {
                codebook_only += 1;
            }
        }
    }
    assert!(level_formats > 0, "no level format in ALL");
    // IQ2_XS and IQ3_XXS are the only codebook-geometry formats left, and each has its decoder
    assert_eq!(codebook_only, 2);
    for format in FOUR {
        assert!(
            levels_of(format).packing().is_some(),
            "{format:?} carries a packing, not just a block shape"
        );
    }
}

#[test]
fn control_one_changed_offset_fails_the_member_comparison() {
    let cases: [(NumericFormat, &str, usize); 4] = [
        (NumericFormat::Iq2Xxs, "sign index", 8),
        (NumericFormat::Iq2S, "grid index high", 8),
        (NumericFormat::Iq1S, "delta sign", 8),
        (NumericFormat::Iq1M, "super scale", 8),
    ];
    for (format, plane, shift) in cases {
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
    let untouched = claims_of(NumericFormat::Iq2S);
    assert_eq!(mismatch(NumericFormat::Iq2S, &untouched), None);
}

#[test]
fn the_crate_describes_these_formats_without_decoding_or_encoding_them() {
    let mut values = [0.0_f32; QK_K];
    for format in FOUR {
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
fn block_layout_is_none_for_the_four_but_their_packing_is_kept() {
    for format in FOUR {
        assert_eq!(format.block_layout(), None, "{format:?}");
        assert!(levels_of(format).packing().is_some(), "{format:?}");
    }
}
