/// Operand width: `W32` selects `wN` registers, `W64` selects `xN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    W32,
    W64,
}

/// Register number 0..=31. Number 31 is `sp` where the instruction takes a
/// base register (`ldr`/`str`) and in the immediate `add`/`sub` forms, and
/// `xzr`/`wzr` in the shifted-register `add`/`sub` operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reg(pub u8);

impl Reg {
    pub const SP: Reg = Reg(31);
    pub const LR: Reg = Reg(30);
}

/// One aarch64 instruction. Field units are named in the field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MachInst {
    Movz {
        width: Width,
        rd: Reg,
        imm16: u16,
        hw: u8,
    },
    Movk {
        width: Width,
        rd: Reg,
        imm16: u16,
        hw: u8,
    },
    /// shifted register, `lsl` by `shift_amount`
    Add {
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        shift_amount: u8,
    },
    Sub {
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        shift_amount: u8,
    },
    /// immediate form, `imm12` shifted left 12 when `shift12`; register 31 is
    /// `sp` for both `rd` and `rn`
    AddImm {
        width: Width,
        rd: Reg,
        rn: Reg,
        imm12: u16,
        shift12: bool,
    },
    SubImm {
        width: Width,
        rd: Reg,
        rn: Reg,
        imm12: u16,
        shift12: bool,
    },
    Ret {
        rn: Reg,
    },
    /// breakpoint trap with the 16-bit comment `imm16`
    Brk {
        imm16: u16,
    },
    /// branch by `offset_words` instructions from this one
    B {
        offset_words: i32,
    },
    /// branch by `offset_words` instructions when `rt` is zero
    Cbz {
        width: Width,
        rt: Reg,
        offset_words: i32,
    },
    /// branch by `offset_words` instructions when `rt` is nonzero
    Cbnz {
        width: Width,
        rt: Reg,
        offset_words: i32,
    },
    /// unsigned immediate offset, `offset` in bytes
    Ldr {
        width: Width,
        rt: Reg,
        rn: Reg,
        offset: u32,
    },
    Str {
        width: Width,
        rt: Reg,
        rn: Reg,
        offset: u32,
    },
    /// `madd rd, rn, rm, xzr`
    Mul {
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    /// `sdiv rd, rn, rm`
    Sdiv {
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    /// `udiv rd, rn, rm`
    Udiv {
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    /// `subs xzr, rn, rm`: sets flags, discards the difference
    Cmp {
        width: Width,
        rn: Reg,
        rm: Reg,
    },
    /// `rd` = 1 when `cond` holds on the flags, else 0 (a w write)
    Cset {
        rd: Reg,
        cond: Cond,
    },
    /// zero-extending byte load, `offset` in bytes
    Ldrb {
        rt: Reg,
        rn: Reg,
        offset: u32,
    },
    Strb {
        rt: Reg,
        rn: Reg,
        offset: u32,
    },
    /// zero-extending halfword load, `offset` in bytes, a multiple of 2
    Ldrh {
        rt: Reg,
        rn: Reg,
        offset: u32,
    },
    Strh {
        rt: Reg,
        rn: Reg,
        offset: u32,
    },
    /// signed bitfield move; `sxtb`, `sxth` and `sxtw` are `immr` 0 with `imms` 7, 15 and 31
    Sbfm {
        width: Width,
        rd: Reg,
        rn: Reg,
        immr: u8,
        imms: u8,
    },
    /// unsigned bitfield move; `uxtb` and `uxth` are `immr` 0 with `imms` 7 and 15
    Ubfm {
        width: Width,
        rd: Reg,
        rn: Reg,
        immr: u8,
        imms: u8,
    },
    /// call: branch by `offset_words` instructions and set x30. Lowering emits 0; the object
    /// writer's `ARM64_RELOC_BRANCH26` carries the real target
    Bl {
        offset_words: i32,
    },
    /// `adrp rd, <page>`: lowering emits page 0; the object writer's `ARM64_RELOC_PAGE21` carries
    /// the real page
    Adrp {
        rd: Reg,
    },
    /// `stp rt, rt2, [rn, #offset]!` of two x registers, `offset` in bytes, a multiple of 8
    StpPre {
        rt: Reg,
        rt2: Reg,
        rn: Reg,
        offset: i32,
    },
    /// `ldp rt, rt2, [rn], #offset` of two x registers, `offset` in bytes, a multiple of 8
    LdpPost {
        rt: Reg,
        rt2: Reg,
        rn: Reg,
        offset: i32,
    },
}

/// aarch64 condition code, the 4-bit `cond` field value in the name's order
/// (ARM ARM C1.2.4 "Condition codes").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cond {
    Eq = 0,
    Ne = 1,
    /// carry set, unsigned `>=`
    Hs = 2,
    /// carry clear, unsigned `<`
    Lo = 3,
    Hi = 8,
    Ls = 9,
    Ge = 10,
    Lt = 11,
    Gt = 12,
    Le = 13,
}

impl Cond {
    /// the condition that holds exactly when `self` does not (flip bit 0)
    #[must_use]
    pub const fn invert(self) -> Cond {
        match self {
            Cond::Eq => Cond::Ne,
            Cond::Ne => Cond::Eq,
            Cond::Hs => Cond::Lo,
            Cond::Lo => Cond::Hs,
            Cond::Hi => Cond::Ls,
            Cond::Ls => Cond::Hi,
            Cond::Ge => Cond::Lt,
            Cond::Lt => Cond::Ge,
            Cond::Gt => Cond::Le,
            Cond::Le => Cond::Gt,
        }
    }
}
