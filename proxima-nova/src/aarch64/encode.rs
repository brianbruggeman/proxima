use thiserror::Error;

use super::inst::{MachInst, Reg, Width};

/// An operand the encoder refuses rather than truncate into a word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum EncodeError {
    #[error("register operand `{operand}` is {value}, must be 0..=31")]
    RegisterOutOfRange { operand: &'static str, value: u8 },
    #[error("operand `{operand}` is {value}, outside {min}..={max}")]
    OutOfRange {
        operand: &'static str,
        value: i64,
        min: i64,
        max: i64,
    },
    #[error("operand `{operand}` is {value}, not a multiple of {align}")]
    Misaligned {
        operand: &'static str,
        value: i64,
        align: i64,
    },
}

fn reg(operand: &'static str, register: Reg) -> Result<u32, EncodeError> {
    match register.0 {
        0..=31 => Ok(u32::from(register.0)),
        value => Err(EncodeError::RegisterOutOfRange { operand, value }),
    }
}

fn ranged(operand: &'static str, value: i64, max: i64) -> Result<u32, EncodeError> {
    match u32::try_from(value) {
        Ok(word) if value <= max => Ok(word),
        _ => Err(EncodeError::OutOfRange {
            operand,
            value,
            min: 0,
            max,
        }),
    }
}

const fn sf(width: Width) -> u32 {
    match width {
        Width::W32 => 0,
        Width::W64 => 1 << 31,
    }
}

// the x form sets N (bit 22) alongside sf, and its bit positions run 0..=63
fn bitfield(
    base: u32,
    width: Width,
    rd: Reg,
    rn: Reg,
    immr: u8,
    imms: u8,
) -> Result<u32, EncodeError> {
    let (max, n_bit) = match width {
        Width::W32 => (31, 0),
        Width::W64 => (63, 1 << 22),
    };
    let immr = ranged("immr", i64::from(immr), max)?;
    let imms = ranged("imms", i64::from(imms), max)?;
    Ok(sf(width) | base | n_bit | immr << 16 | imms << 10 | reg("rn", rn)? << 5 | reg("rd", rd)?)
}

fn wide_move(base: u32, width: Width, rd: Reg, imm16: u16, hw: u8) -> Result<u32, EncodeError> {
    let max_hw = match width {
        Width::W32 => 1,
        Width::W64 => 3,
    };
    let hw = ranged("hw", i64::from(hw), max_hw)?;
    Ok(sf(width) | base | hw << 21 | u32::from(imm16) << 5 | reg("rd", rd)?)
}

fn shifted(
    base: u32,
    width: Width,
    rd: Reg,
    rn: Reg,
    rm: Reg,
    amount: u8,
) -> Result<u32, EncodeError> {
    let max = match width {
        Width::W32 => 31,
        Width::W64 => 63,
    };
    let amount = ranged("shift_amount", i64::from(amount), max)?;
    Ok(sf(width)
        | base
        | reg("rm", rm)? << 16
        | amount << 10
        | reg("rn", rn)? << 5
        | reg("rd", rd)?)
}

// register 31 is sp for rd and rn in the immediate forms
fn immediate(
    base: u32,
    width: Width,
    rd: Reg,
    rn: Reg,
    imm12: u16,
    shift12: bool,
) -> Result<u32, EncodeError> {
    let imm12 = ranged("imm12", i64::from(imm12), 4095)?;
    Ok(sf(width)
        | base
        | u32::from(shift12) << 22
        | imm12 << 10
        | reg("rn", rn)? << 5
        | reg("rd", rd)?)
}

const fn access_size(width: Width) -> u32 {
    match width {
        Width::W32 => 4,
        Width::W64 => 8,
    }
}

// size is the access in bytes (1, 2, 4, 8); it selects the size field and scales the offset
fn memory(base: u32, size: u32, rt: Reg, rn: Reg, offset: u32) -> Result<u32, EncodeError> {
    if !offset.is_multiple_of(size) {
        return Err(EncodeError::Misaligned {
            operand: "offset",
            value: i64::from(offset),
            align: i64::from(size),
        });
    }
    let max = 4095 * i64::from(size);
    if i64::from(offset) > max {
        return Err(EncodeError::OutOfRange {
            operand: "offset",
            value: i64::from(offset),
            min: 0,
            max,
        });
    }
    let scaled = offset / size;
    let size_bits = size.trailing_zeros() << 30;
    Ok(size_bits | base | scaled << 10 | reg("rn", rn)? << 5 | reg("rt", rt)?)
}

fn compare_branch(base: u32, width: Width, rt: Reg, offset_words: i32) -> Result<u32, EncodeError> {
    let value = i64::from(offset_words);
    if !(-(1 << 18)..(1 << 18)).contains(&value) {
        return Err(EncodeError::OutOfRange {
            operand: "offset_words",
            value,
            min: -(1 << 18),
            max: (1 << 18) - 1,
        });
    }
    Ok(sf(width) | base | (offset_words as u32 & 0x0007_ffff) << 5 | reg("rt", rt)?)
}

fn branch26(base: u32, offset_words: i32) -> Result<u32, EncodeError> {
    let value = i64::from(offset_words);
    if !(-(1 << 25)..(1 << 25)).contains(&value) {
        return Err(EncodeError::OutOfRange {
            operand: "offset_words",
            value,
            min: -(1 << 25),
            max: (1 << 25) - 1,
        });
    }
    Ok(base | (offset_words as u32 & 0x03ff_ffff))
}

// 64-bit register pair, signed imm7 scaled by 8
fn pair(base: u32, rt: Reg, rt2: Reg, rn: Reg, offset: i32) -> Result<u32, EncodeError> {
    if offset % 8 != 0 {
        return Err(EncodeError::Misaligned {
            operand: "offset",
            value: i64::from(offset),
            align: 8,
        });
    }
    let scaled = offset / 8;
    if !(-64..64).contains(&scaled) {
        return Err(EncodeError::OutOfRange {
            operand: "offset",
            value: i64::from(offset),
            min: -512,
            max: 504,
        });
    }
    Ok(base
        | (scaled as u32 & 0x7f) << 15
        | reg("rt2", rt2)? << 10
        | reg("rn", rn)? << 5
        | reg("rt", rt)?)
}

/// Encode one instruction into its 32-bit little-endian word.
#[must_use = "the encoded word or the refusal must be handled"]
pub fn encode(inst: MachInst) -> Result<u32, EncodeError> {
    match inst {
        MachInst::Movz {
            width,
            rd,
            imm16,
            hw,
        } => wide_move(0x5280_0000, width, rd, imm16, hw),
        MachInst::Movk {
            width,
            rd,
            imm16,
            hw,
        } => wide_move(0x7280_0000, width, rd, imm16, hw),
        MachInst::Add {
            width,
            rd,
            rn,
            rm,
            shift_amount,
        } => shifted(0x0b00_0000, width, rd, rn, rm, shift_amount),
        MachInst::Sub {
            width,
            rd,
            rn,
            rm,
            shift_amount,
        } => shifted(0x4b00_0000, width, rd, rn, rm, shift_amount),
        MachInst::AddImm {
            width,
            rd,
            rn,
            imm12,
            shift12,
        } => immediate(0x1100_0000, width, rd, rn, imm12, shift12),
        MachInst::SubImm {
            width,
            rd,
            rn,
            imm12,
            shift12,
        } => immediate(0x5100_0000, width, rd, rn, imm12, shift12),
        MachInst::Ret { rn } => Ok(0xd65f_0000 | reg("rn", rn)? << 5),
        MachInst::Brk { imm16 } => Ok(0xd420_0000 | u32::from(imm16) << 5),
        MachInst::B { offset_words } => branch26(0x1400_0000, offset_words),
        MachInst::Bl { offset_words } => branch26(0x9400_0000, offset_words),
        MachInst::Adrp { rd } => Ok(0x9000_0000 | reg("rd", rd)?),
        MachInst::StpPre {
            rt,
            rt2,
            rn,
            offset,
        } => pair(0xa980_0000, rt, rt2, rn, offset),
        MachInst::LdpPost {
            rt,
            rt2,
            rn,
            offset,
        } => pair(0xa8c0_0000, rt, rt2, rn, offset),
        MachInst::Cbz {
            width,
            rt,
            offset_words,
        } => compare_branch(0x3400_0000, width, rt, offset_words),
        MachInst::Cbnz {
            width,
            rt,
            offset_words,
        } => compare_branch(0x3500_0000, width, rt, offset_words),
        MachInst::Ldr {
            width,
            rt,
            rn,
            offset,
        } => memory(0x3940_0000, access_size(width), rt, rn, offset),
        MachInst::Str {
            width,
            rt,
            rn,
            offset,
        } => memory(0x3900_0000, access_size(width), rt, rn, offset),
        MachInst::Ldrb { rt, rn, offset } => memory(0x3940_0000, 1, rt, rn, offset),
        MachInst::Strb { rt, rn, offset } => memory(0x3900_0000, 1, rt, rn, offset),
        MachInst::Ldrh { rt, rn, offset } => memory(0x3940_0000, 2, rt, rn, offset),
        MachInst::Strh { rt, rn, offset } => memory(0x3900_0000, 2, rt, rn, offset),
        // madd with ra = 31 (xzr)
        MachInst::Mul { width, rd, rn, rm } => Ok(sf(width)
            | 0x1b00_0000
            | reg("rm", rm)? << 16
            | 31 << 10
            | reg("rn", rn)? << 5
            | reg("rd", rd)?),
        MachInst::Sdiv { width, rd, rn, rm } => Ok(sf(width)
            | 0x1ac0_0c00
            | reg("rm", rm)? << 16
            | reg("rn", rn)? << 5
            | reg("rd", rd)?),
        MachInst::Udiv { width, rd, rn, rm } => Ok(sf(width)
            | 0x1ac0_0800
            | reg("rm", rm)? << 16
            | reg("rn", rn)? << 5
            | reg("rd", rd)?),
        // subs with rd = 31 (xzr)
        MachInst::Cmp { width, rn, rm } => shifted(0x6b00_0000, width, Reg(31), rn, rm, 0),
        MachInst::Sbfm {
            width,
            rd,
            rn,
            immr,
            imms,
        } => bitfield(0x1300_0000, width, rd, rn, immr, imms),
        MachInst::Ubfm {
            width,
            rd,
            rn,
            immr,
            imms,
        } => bitfield(0x5300_0000, width, rd, rn, immr, imms),
        // csinc rd, wzr, wzr, !cond
        MachInst::Cset { rd, cond } => {
            Ok(0x1a9f_07e0 | (cond.invert() as u32) << 12 | reg("rd", rd)?)
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::fmt;
    use std::fs;
    use std::process::Command;
    use std::vec::Vec;

    use object::{Object, ObjectSection, ObjectSymbol};

    use super::super::inst::Cond;
    use super::*;

    const SYMBOL: &str = "_nova_listing";

    const LISTING: &str = "\
    ret
    movz x0, #42
    movz w1, #7, lsl #16
    movk x0, #1, lsl #48
    movk w1, #9, lsl #16
    add x2, x0, x1
    add w3, w1, w1
    add x9, x0, x1, lsl #3
    sub x4, x2, x0
    sub w5, w3, w1
    b 1f
    cbz x9, 1f
    cbnz w10, 1f
    mul x9, x10, x11
    mul w9, w10, w11
    cmp x9, x10
    cmp w9, w10
    cset w9, lt
    cset w9, hi
    ldrb w9, [x10, #1]
    strb w9, [x10, #3]
    ldrh w9, [x10, #2]
    strh w9, [x10, #6]
    sxtb w7, w8
    sxtb x9, w10
    sxth w11, w12
    sxth x13, w14
    sxtw x3, w4
    uxtb w5, w6
    uxth w15, w0
    bl nova_callee
    stp x29, x30, [sp, #-16]!
    ldp x29, x30, [sp], #16
    stp x19, x20, [sp, #-16]!
    ldp x19, x20, [sp], #16
    brk #0
1:
    ldr x5, [sp, #16]
    ldr w6, [x0, #4]
    str x7, [sp, #8]
    str w8, [x1, #8]
    sub sp, sp, #32
    add sp, sp, #32
    add x10, sp, #16
    adrp x5, nova_data@PAGE
    add x5, x5, nova_data@PAGEOFF
    movz x16, #5
2:
    sub sp, sp, #1, lsl #12
    ldr xzr, [sp]
    sub x16, x16, #1
    cbnz x16, 2b
    add sp, sp, #2, lsl #12
    add x0, sp, #1, lsl #12
    add x0, x0, #936
    add x9, x9, #4095, lsl #12
";

    fn instructions() -> Vec<MachInst> {
        let (x, w) = (Width::W64, Width::W32);
        let r = Reg;
        std::vec![
            MachInst::Ret { rn: Reg::LR },
            MachInst::Movz {
                width: x,
                rd: r(0),
                imm16: 42,
                hw: 0
            },
            MachInst::Movz {
                width: w,
                rd: r(1),
                imm16: 7,
                hw: 1
            },
            MachInst::Movk {
                width: x,
                rd: r(0),
                imm16: 1,
                hw: 3
            },
            MachInst::Movk {
                width: w,
                rd: r(1),
                imm16: 9,
                hw: 1
            },
            MachInst::Add {
                width: x,
                rd: r(2),
                rn: r(0),
                rm: r(1),
                shift_amount: 0
            },
            MachInst::Add {
                width: w,
                rd: r(3),
                rn: r(1),
                rm: r(1),
                shift_amount: 0
            },
            MachInst::Add {
                width: x,
                rd: r(9),
                rn: r(0),
                rm: r(1),
                shift_amount: 3
            },
            MachInst::Sub {
                width: x,
                rd: r(4),
                rn: r(2),
                rm: r(0),
                shift_amount: 0
            },
            MachInst::Sub {
                width: w,
                rd: r(5),
                rn: r(3),
                rm: r(1),
                shift_amount: 0
            },
            MachInst::B { offset_words: 26 },
            MachInst::Cbz {
                width: x,
                rt: r(9),
                offset_words: 25
            },
            MachInst::Cbnz {
                width: w,
                rt: r(10),
                offset_words: 24
            },
            MachInst::Mul {
                width: x,
                rd: r(9),
                rn: r(10),
                rm: r(11)
            },
            MachInst::Mul {
                width: w,
                rd: r(9),
                rn: r(10),
                rm: r(11)
            },
            MachInst::Cmp {
                width: x,
                rn: r(9),
                rm: r(10)
            },
            MachInst::Cmp {
                width: w,
                rn: r(9),
                rm: r(10)
            },
            MachInst::Cset {
                rd: r(9),
                cond: Cond::Lt
            },
            MachInst::Cset {
                rd: r(9),
                cond: Cond::Hi
            },
            MachInst::Ldrb {
                rt: r(9),
                rn: r(10),
                offset: 1
            },
            MachInst::Strb {
                rt: r(9),
                rn: r(10),
                offset: 3
            },
            MachInst::Ldrh {
                rt: r(9),
                rn: r(10),
                offset: 2
            },
            MachInst::Strh {
                rt: r(9),
                rn: r(10),
                offset: 6
            },
            MachInst::Sbfm {
                width: w,
                rd: r(7),
                rn: r(8),
                immr: 0,
                imms: 7
            },
            MachInst::Sbfm {
                width: x,
                rd: r(9),
                rn: r(10),
                immr: 0,
                imms: 7
            },
            MachInst::Sbfm {
                width: w,
                rd: r(11),
                rn: r(12),
                immr: 0,
                imms: 15
            },
            MachInst::Sbfm {
                width: x,
                rd: r(13),
                rn: r(14),
                immr: 0,
                imms: 15
            },
            MachInst::Sbfm {
                width: x,
                rd: r(3),
                rn: r(4),
                immr: 0,
                imms: 31
            },
            MachInst::Ubfm {
                width: w,
                rd: r(5),
                rn: r(6),
                immr: 0,
                imms: 7
            },
            MachInst::Ubfm {
                width: w,
                rd: r(15),
                rn: r(0),
                immr: 0,
                imms: 15
            },
            MachInst::Bl { offset_words: 0 },
            MachInst::StpPre {
                rt: r(29),
                rt2: r(30),
                rn: Reg::SP,
                offset: -16
            },
            MachInst::LdpPost {
                rt: r(29),
                rt2: r(30),
                rn: Reg::SP,
                offset: 16
            },
            MachInst::StpPre {
                rt: r(19),
                rt2: r(20),
                rn: Reg::SP,
                offset: -16
            },
            MachInst::LdpPost {
                rt: r(19),
                rt2: r(20),
                rn: Reg::SP,
                offset: 16
            },
            MachInst::Brk { imm16: 0 },
            MachInst::Ldr {
                width: x,
                rt: r(5),
                rn: Reg::SP,
                offset: 16
            },
            MachInst::Ldr {
                width: w,
                rt: r(6),
                rn: r(0),
                offset: 4
            },
            MachInst::Str {
                width: x,
                rt: r(7),
                rn: Reg::SP,
                offset: 8
            },
            MachInst::Str {
                width: w,
                rt: r(8),
                rn: r(1),
                offset: 8
            },
            MachInst::SubImm {
                width: x,
                rd: Reg::SP,
                rn: Reg::SP,
                imm12: 32,
                shift12: false
            },
            MachInst::AddImm {
                width: x,
                rd: Reg::SP,
                rn: Reg::SP,
                imm12: 32,
                shift12: false
            },
            MachInst::AddImm {
                width: x,
                rd: r(10),
                rn: Reg::SP,
                imm12: 16,
                shift12: false
            },
            MachInst::Adrp { rd: r(5) },
            MachInst::AddImm {
                width: x,
                rd: r(5),
                rn: r(5),
                imm12: 0,
                shift12: false
            },
            MachInst::Movz {
                width: x,
                rd: r(16),
                imm16: 5,
                hw: 0
            },
            MachInst::SubImm {
                width: x,
                rd: Reg::SP,
                rn: Reg::SP,
                imm12: 1,
                shift12: true
            },
            MachInst::Ldr {
                width: x,
                rt: Reg(31),
                rn: Reg::SP,
                offset: 0
            },
            MachInst::SubImm {
                width: x,
                rd: r(16),
                rn: r(16),
                imm12: 1,
                shift12: false
            },
            MachInst::Cbnz {
                width: x,
                rt: r(16),
                offset_words: -3
            },
            MachInst::AddImm {
                width: x,
                rd: Reg::SP,
                rn: Reg::SP,
                imm12: 2,
                shift12: true
            },
            MachInst::AddImm {
                width: x,
                rd: r(0),
                rn: Reg::SP,
                imm12: 1,
                shift12: true
            },
            MachInst::AddImm {
                width: x,
                rd: r(0),
                rn: r(0),
                imm12: 936,
                shift12: false
            },
            MachInst::AddImm {
                width: x,
                rd: r(9),
                rn: r(9),
                imm12: 4095,
                shift12: true
            },
        ]
    }

    struct Comparison {
        mismatched: usize,
        of: usize,
    }

    impl fmt::Display for Comparison {
        fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(out, "mismatched={} of={}", self.mismatched, self.of)
        }
    }

    fn compare(left: &[u32], right: &[u32]) -> Comparison {
        let of = left.len().max(right.len());
        let mismatched = (0..of)
            .filter(|&index| left.get(index) != right.get(index))
            .count();
        Comparison { mismatched, of }
    }

    fn assembler_words() -> Vec<u32> {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("listing.s");
        let object_path = dir.path().join("listing.o");
        let text = std::format!(".text\n.globl {SYMBOL}\n.p2align 2\n{SYMBOL}:\n{LISTING}");
        fs::write(&source, text).expect("write listing");
        let status = Command::new("cc")
            .args(["-c", "-arch", "arm64", "-o"])
            .arg(&object_path)
            .arg(&source)
            .status()
            .expect("spawn cc");
        assert!(status.success(), "cc -c failed");
        let bytes = fs::read(&object_path).expect("read object");
        let file = object::File::parse(&*bytes).expect("parse object");
        let symbol = file
            .symbols()
            .find(|symbol| symbol.name() == Ok(SYMBOL))
            .expect("global symbol");
        let section = file
            .section_by_index(symbol.section_index().expect("symbol section"))
            .expect("section");
        assert_eq!(section.name(), Ok("__text"));
        let data = section.data().expect("section data");
        let start = (symbol.address() - section.address()) as usize;
        data[start..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect()
    }

    fn nova_words() -> Vec<u32> {
        instructions()
            .into_iter()
            .map(|inst| encode(inst).expect("encodable"))
            .collect()
    }

    fn listing_count() -> usize {
        LISTING.lines().filter(|line| !line.ends_with(':')).count()
    }

    #[test]
    fn sdiv_w0_w1_w2_is_1ac20c20() {
        let inst = MachInst::Sdiv {
            width: Width::W32,
            rd: Reg(0),
            rn: Reg(1),
            rm: Reg(2),
        };
        assert_eq!(encode(inst), Ok(0x1ac2_0c20));
    }

    #[test]
    fn udiv_x0_x1_x2_is_9ac20820() {
        let inst = MachInst::Udiv {
            width: Width::W64,
            rd: Reg(0),
            rn: Reg(1),
            rm: Reg(2),
        };
        assert_eq!(encode(inst), Ok(0x9ac2_0820));
    }

    #[test]
    fn ret_is_d65f03c0() {
        assert_eq!(encode(MachInst::Ret { rn: Reg::LR }), Ok(0xd65f_03c0));
    }

    #[test]
    fn refuses_misaligned_ldr_offset() {
        let inst = MachInst::Ldr {
            width: Width::W64,
            rt: Reg(0),
            rn: Reg::SP,
            offset: 4,
        };
        assert_eq!(
            encode(inst),
            Err(EncodeError::Misaligned {
                operand: "offset",
                value: 4,
                align: 8
            })
        );
    }

    #[test]
    fn assembler_words_against_themselves_are_clean() {
        let words = assembler_words();
        assert_eq!(words.len(), listing_count());
        assert_eq!(words[0], 0xd65f_03c0);
        let comparison = compare(&words, &words);
        std::println!("{comparison} K={}", words.len());
        assert_eq!((comparison.mismatched, comparison.of), (0, words.len()));
    }

    #[test]
    fn matches_system_assembler() {
        let comparison = compare(&nova_words(), &assembler_words());
        std::println!("{comparison}");
        assert_eq!((comparison.mismatched, comparison.of), (0, listing_count()));
    }

    #[test]
    fn comparison_flags_one_flipped_bit() {
        let mut flipped = nova_words();
        flipped[3] ^= 1;
        let comparison = compare(&flipped, &assembler_words());
        std::println!("{comparison}");
        assert_eq!((comparison.mismatched, comparison.of), (1, listing_count()));
    }
}
