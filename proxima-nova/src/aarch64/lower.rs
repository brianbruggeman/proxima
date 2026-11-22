//! Lower a nova [`Function`] to aarch64 machine words.
//!
//! Composes [`verify`] (refuse bad IR first), [`MachInst`] (the payload) and
//! [`encode`] (the word). [`crate::pipeline`] wraps this as a pipe stage.
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use thiserror::Error;

use super::parallel_move::resolve;
use super::{Allocation, Cond, EncodeError, MachInst, Reg, RegAllocError, Width, allocate, encode};
use crate::ir::{
    BlockId, Cond as IrCond, Function, Inst, Slot, Terminator, Type, Value, VerifyError, verify,
};
use crate::pipeline::{Relocation, RelocationKind};

/// Why a function was refused rather than lowered.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LowerError {
    #[error("invalid ir: {0}")]
    Verify(#[from] VerifyError),
    #[error("cannot encode: {0}")]
    Encode(#[from] EncodeError),
    #[error("unsupported instruction `{inst}`")]
    Unsupported { inst: &'static str },
    #[error("unsupported `{inst}` of type {ty:?}")]
    UnsupportedType { inst: &'static str, ty: Type },
    #[error("`{inst}` cannot change width from {from:?} to {to:?}")]
    BadWidthChange {
        inst: &'static str,
        from: Type,
        to: Type,
    },
    #[error("register assignment: {0}")]
    Regalloc(#[from] RegAllocError),
    #[error("jump to block {target:?} passes {args} args to {params} params")]
    ArgCount {
        target: BlockId,
        args: usize,
        params: usize,
    },
    #[error("stack slot align {align} must be a power of two no larger than {FRAME_ALIGN}")]
    BadSlotAlign { align: u32 },
    #[error("call to `{callee}` passes {args} args, only {MAX_CALL_ARGS} go in x0..x7")]
    CallArgs { callee: String, args: usize },
}

/// A lowered function: its instructions and the sites that still need a relocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lowered {
    pub insts: Vec<MachInst>,
    /// every instruction that names a symbol: `bl` (Branch26), `adrp` (Page21) and the
    /// page-offset `add` (PageOff12); `offset` is in bytes from the start of `insts`
    pub relocations: Vec<Relocation>,
}

const MAX_CALL_ARGS: usize = 8;
const FRAME_POINTER: Reg = Reg(29);
const RETURN_REG: Reg = Reg(0);
const XZR: Reg = Reg(31);
const WIDTH: Width = Width::W64;
const FRAME_ALIGN: u32 = 16;
const MAX_IMM12: u32 = 4095;
const MAX_OFFSET: u32 = (MAX_IMM12 << 12) | MAX_IMM12;
const PROBE_FLOOR: u32 = 1024;
const UNROLLED_PAGES: u32 = 4;
const LOOP_COUNTER: Reg = Reg(16);

// registers come from [`allocate`]: x0..x15 by liveness, entry params precolored to x0..x7, every
// value 64-bit, the return value moves to x0; no spilling.
//
// I32 values live in the low half of those registers. add/sub carry only
// upward, so the low 32 bits of a 64-bit add/sub depend on the low 32 bits of
// its operands alone; garbage above bit 31 (a w-register argument leaves x0's
// upper half undefined) never reaches them. Consumers read only the low half:
// `str w`, and the `w0` the C runtime takes as the exit status.
// register move is `add rd, rn, xzr`; rm = 31 is xzr in the shifted-register form
fn mov(rd: Reg, rn: Reg) -> MachInst {
    MachInst::Add {
        width: WIDTH,
        rd,
        rn,
        rm: XZR,
        shift_amount: 0,
    }
}

// a w-form write clears bits 32..63, which an I32 value may leave holding garbage
fn mov_w(rd: Reg, rn: Reg) -> MachInst {
    MachInst::Add {
        width: Width::W32,
        rd,
        rn,
        rm: XZR,
        shift_amount: 0,
    }
}

fn ubfm_w(rd: Reg, rn: Reg, imms: u8) -> MachInst {
    MachInst::Ubfm {
        width: Width::W32,
        rd,
        rn,
        immr: 0,
        imms,
    }
}

const fn bad_width_change(inst: &'static str, from: Type, to: Type) -> LowerError {
    LowerError::BadWidthChange { inst, from, to }
}

fn width_change(
    ctx: &Ctx,
    result: Option<Value>,
    value: Value,
    inst: &'static str,
) -> Result<(Reg, Reg, Type), LowerError> {
    let rd = result_reg(ctx, result, inst)?;
    Ok((rd, ctx.reg(value)?, ctx.type_of(value)?))
}

// movz the low chunk (zero included), movk every other non-zero chunk
fn materialize(rd: Reg, imm: i64, out: &mut Vec<MachInst>) {
    let bits = imm as u64;
    out.push(MachInst::Movz {
        width: WIDTH,
        rd,
        imm16: bits as u16,
        hw: 0,
    });
    for hw in 1..4u8 {
        let imm16 = (bits >> (16 * u32::from(hw))) as u16;
        if imm16 != 0 {
            out.push(MachInst::Movk {
                width: WIDTH,
                rd,
                imm16,
                hw,
            });
        }
    }
}

// the stack frame: slot offsets in order of appearance, total rounded to 16
struct Frame {
    offsets: Vec<u32>,
    size: u32,
}

fn to_u32(operand: &'static str, value: u64) -> Result<u32, LowerError> {
    u32::try_from(value).map_err(|_| {
        EncodeError::OutOfRange {
            operand,
            value: i64::try_from(value).unwrap_or(i64::MAX),
            min: 0,
            max: i64::from(u32::MAX),
        }
        .into()
    })
}

// each slot sits at the next multiple of its align; the frame is the end of
// the last slot rounded up to FRAME_ALIGN so sp stays 16-byte aligned
fn layout(function: &Function) -> Result<Frame, LowerError> {
    let mut cursor = 0u64;
    let mut offsets = Vec::new();
    for Slot { size, align } in &function.slots {
        if !align.is_power_of_two() || *align > FRAME_ALIGN {
            return Err(LowerError::BadSlotAlign { align: *align });
        }
        let offset = cursor.next_multiple_of(u64::from(*align));
        offsets.push(to_u32("slot_offset", offset)?);
        cursor = offset + u64::from(*size);
    }
    let size = to_u32("frame", cursor.next_multiple_of(u64::from(FRAME_ALIGN)))?;
    Ok(Frame { offsets, size })
}

// a value is `hi << 12 | lo`, each 12 bits
fn check_offset(operand: &'static str, value: u32) -> Result<(), LowerError> {
    if value > MAX_OFFSET {
        return Err(EncodeError::OutOfRange {
            operand,
            value: i64::from(value),
            min: 0,
            max: i64::from(MAX_OFFSET),
        }
        .into());
    }
    Ok(())
}

fn imm12_of(value: u32) -> u16 {
    u16::try_from(value & MAX_IMM12).unwrap_or(u16::MAX)
}

fn add_imm(rd: Reg, rn: Reg, imm12: u16, shift12: bool) -> MachInst {
    MachInst::AddImm {
        width: WIDTH,
        rd,
        rn,
        imm12,
        shift12,
    }
}

fn sub_imm(rd: Reg, rn: Reg, imm12: u16, shift12: bool) -> MachInst {
    MachInst::SubImm {
        width: WIDTH,
        rd,
        rn,
        imm12,
        shift12,
    }
}

// `ldr xzr, [sp]` touches the page sp now points into
fn probe() -> MachInst {
    MachInst::Ldr {
        width: WIDTH,
        rt: XZR,
        rn: Reg::SP,
        offset: 0,
    }
}

// x16 counts the pages down; the prologue runs before any body use of x16 (R28)
fn page_loop(pages: u32) -> [MachInst; 5] {
    [
        MachInst::Movz {
            width: WIDTH,
            rd: LOOP_COUNTER,
            imm16: imm12_of(pages),
            hw: 0,
        },
        sub_imm(Reg::SP, Reg::SP, 1, true),
        probe(),
        sub_imm(LOOP_COUNTER, LOOP_COUNTER, 1, false),
        MachInst::Cbnz {
            width: WIDTH,
            rt: LOOP_COUNTER,
            offset_words: -3,
        },
    ]
}

// pages probed in order from the old sp down, as LLVM does: unrolled up to 4 pages, a loop beyond;
// the remainder is probed only above 1024 bytes
fn grow_frame(frame: u32) -> Result<Vec<MachInst>, LowerError> {
    check_offset("frame", frame)?;
    let (pages, rest) = (frame >> 12, frame & MAX_IMM12);
    let mut out = Vec::new();
    if pages > UNROLLED_PAGES {
        out.extend(page_loop(pages));
    } else {
        for _ in 0..pages {
            out.push(sub_imm(Reg::SP, Reg::SP, 1, true));
            out.push(probe());
        }
    }
    if rest > 0 {
        out.push(sub_imm(Reg::SP, Reg::SP, imm12_of(rest), false));
        if rest > PROBE_FLOOR {
            out.push(probe());
        }
    }
    Ok(out)
}

fn release_frame(frame: u32) -> Result<Vec<MachInst>, LowerError> {
    check_offset("frame", frame)?;
    let (pages, rest) = (frame >> 12, frame & MAX_IMM12);
    let mut out = Vec::new();
    if pages > 0 {
        out.push(add_imm(Reg::SP, Reg::SP, imm12_of(pages), true));
    }
    if rest > 0 {
        out.push(add_imm(Reg::SP, Reg::SP, imm12_of(rest), false));
    }
    Ok(out)
}

// one add when the offset is imm12 or a page multiple, else `hi << 12` then `lo`, the second add reading rd
fn slot_address(rd: Reg, offset: u32, out: &mut Vec<MachInst>) -> Result<(), LowerError> {
    check_offset("slot_offset", offset)?;
    let (hi, lo) = (offset >> 12, offset & MAX_IMM12);
    match (hi, lo) {
        (_, 0) if hi > 0 => out.push(add_imm(rd, Reg::SP, imm12_of(hi), true)),
        (0, _) => out.push(add_imm(rd, Reg::SP, imm12_of(lo), false)),
        _ => {
            out.push(add_imm(rd, Reg::SP, imm12_of(hi), true));
            out.push(add_imm(rd, rd, imm12_of(lo), false));
        }
    }
    Ok(())
}

// narrow types occupy the low bits of a register, so they test and compare as w
const fn value_width(ty: Type) -> Width {
    match ty {
        Type::I8 | Type::I16 | Type::I32 => Width::W32,
        Type::I64 | Type::Ptr => Width::W64,
    }
}

// signed conds map to ge/lt/gt/le, unsigned to hs/lo/hi/ls
const fn machine_cond(cond: IrCond) -> Cond {
    match cond {
        IrCond::Eq => Cond::Eq,
        IrCond::Ne => Cond::Ne,
        IrCond::Slt => Cond::Lt,
        IrCond::Sle => Cond::Le,
        IrCond::Sgt => Cond::Gt,
        IrCond::Sge => Cond::Ge,
        IrCond::Ult => Cond::Lo,
        IrCond::Ule => Cond::Ls,
        IrCond::Ugt => Cond::Hi,
        IrCond::Uge => Cond::Hs,
    }
}

fn load_inst(ty: Type, rt: Reg, rn: Reg, offset: u32) -> MachInst {
    match ty {
        Type::I8 => MachInst::Ldrb { rt, rn, offset },
        Type::I16 => MachInst::Ldrh { rt, rn, offset },
        Type::I32 | Type::I64 | Type::Ptr => MachInst::Ldr {
            width: value_width(ty),
            rt,
            rn,
            offset,
        },
    }
}

fn store_inst(ty: Type, rt: Reg, rn: Reg, offset: u32) -> MachInst {
    match ty {
        Type::I8 => MachInst::Strb { rt, rn, offset },
        Type::I16 => MachInst::Strh { rt, rn, offset },
        Type::I32 | Type::I64 | Type::Ptr => MachInst::Str {
            width: value_width(ty),
            rt,
            rn,
            offset,
        },
    }
}

struct Ctx {
    allocation: Allocation,
    types: BTreeMap<Value, Type>,
    frame: Frame,
    relocations: Vec<Relocation>,
}

impl Ctx {
    // `at` is an index into the emitted instructions
    fn relocate(&mut self, at: usize, kind: RelocationKind, symbol: &str) {
        self.relocations.push(Relocation {
            offset: at as u64 * 4,
            kind,
            symbol: symbol.into(),
        });
    }

    fn reg(&self, value: Value) -> Result<Reg, LowerError> {
        Ok(self.allocation.reg(value)?)
    }

    fn type_of(&self, value: Value) -> Result<Type, LowerError> {
        self.types
            .get(&value)
            .copied()
            .ok_or(LowerError::Unsupported {
                inst: "value without a known type",
            })
    }

    // arithmetic and compares read whole registers; I8/I16 results leave the upper bits
    // unnormalised, so they are refused rather than compared on garbage
    fn arith_type(&self, value: Value, inst: &'static str) -> Result<Type, LowerError> {
        match self.type_of(value)? {
            ty @ (Type::I8 | Type::I16) => Err(LowerError::UnsupportedType { inst, ty }),
            ty => Ok(ty),
        }
    }

    fn define(&mut self, result: Option<Value>, ty: Type) {
        if let Some(value) = result {
            self.types.insert(value, ty);
        }
    }
}

fn result_reg(ctx: &Ctx, result: Option<Value>, inst: &'static str) -> Result<Reg, LowerError> {
    ctx.reg(result.ok_or(LowerError::Unsupported { inst })?)
}

fn lower_inst(
    inst: &Inst,
    result: Option<Value>,
    ctx: &mut Ctx,
    out: &mut Vec<MachInst>,
) -> Result<(), LowerError> {
    match inst {
        Inst::Iconst { ty, imm } => {
            materialize(result_reg(ctx, result, "Iconst without result")?, *imm, out);
            ctx.define(result, *ty);
        }
        Inst::Iadd(lhs, rhs) | Inst::Isub(lhs, rhs) => {
            let rd = result_reg(ctx, result, "Iadd/Isub without result")?;
            let name = if matches!(inst, Inst::Iadd(..)) {
                "Iadd"
            } else {
                "Isub"
            };
            let ty = ctx.arith_type(*lhs, name)?;
            ctx.arith_type(*rhs, name)?;
            let (rn, rm) = (ctx.reg(*lhs)?, ctx.reg(*rhs)?);
            let width = WIDTH;
            out.push(if matches!(inst, Inst::Iadd(..)) {
                MachInst::Add {
                    width,
                    rd,
                    rn,
                    rm,
                    shift_amount: 0,
                }
            } else {
                MachInst::Sub {
                    width,
                    rd,
                    rn,
                    rm,
                    shift_amount: 0,
                }
            });
            ctx.define(result, ty);
        }
        Inst::SlotAddr(slot) => {
            let rd = result_reg(ctx, result, "SlotAddr without result")?;
            let offset = ctx
                .frame
                .offsets
                .get(slot.0 as usize)
                .copied()
                .ok_or(LowerError::Unsupported { inst: "SlotAddr" })?;
            slot_address(rd, offset, out)?;
            ctx.define(result, Type::Ptr);
        }
        Inst::DataAddr { symbol } => {
            let rd = result_reg(ctx, result, "DataAddr without result")?;
            ctx.relocate(out.len(), RelocationKind::Page21, symbol);
            out.push(MachInst::Adrp { rd });
            ctx.relocate(out.len(), RelocationKind::PageOff12, symbol);
            out.push(MachInst::AddImm {
                width: WIDTH,
                rd,
                rn: rd,
                imm12: 0,
                shift12: false,
            });
            ctx.define(result, Type::Ptr);
        }
        Inst::Imul(lhs, rhs) => {
            let rd = result_reg(ctx, result, "Imul without result")?;
            let ty = ctx.arith_type(*lhs, "Imul")?;
            ctx.arith_type(*rhs, "Imul")?;
            out.push(MachInst::Mul {
                width: value_width(ty),
                rd,
                rn: ctx.reg(*lhs)?,
                rm: ctx.reg(*rhs)?,
            });
            ctx.define(result, ty);
        }
        Inst::Sdiv(lhs, rhs) | Inst::Udiv(lhs, rhs) => {
            let name = if matches!(inst, Inst::Sdiv(..)) {
                "Sdiv"
            } else {
                "Udiv"
            };
            let rd = result_reg(ctx, result, "division without result")?;
            let ty = ctx.arith_type(*lhs, name)?;
            ctx.arith_type(*rhs, name)?;
            let (width, rn, rm) = (value_width(ty), ctx.reg(*lhs)?, ctx.reg(*rhs)?);
            out.push(if matches!(inst, Inst::Sdiv(..)) {
                MachInst::Sdiv { width, rd, rn, rm }
            } else {
                MachInst::Udiv { width, rd, rn, rm }
            });
            ctx.define(result, ty);
        }
        Inst::Icmp { cond, a, b } => {
            let rd = result_reg(ctx, result, "Icmp without result")?;
            let ty = ctx.arith_type(*a, "Icmp")?;
            ctx.arith_type(*b, "Icmp")?;
            out.push(MachInst::Cmp {
                width: value_width(ty),
                rn: ctx.reg(*a)?,
                rm: ctx.reg(*b)?,
            });
            out.push(MachInst::Cset {
                rd,
                cond: machine_cond(*cond),
            });
            ctx.define(result, Type::I8);
        }
        Inst::Sextend { ty, value } => {
            let (rd, rn, from) = width_change(ctx, result, *value, "Sextend")?;
            let imms = match (from, *ty) {
                (Type::I8, Type::I32 | Type::I64) => 7,
                (Type::I16, Type::I32 | Type::I64) => 15,
                (Type::I32, Type::I64) => 31,
                _ => return Err(bad_width_change("Sextend", from, *ty)),
            };
            out.push(MachInst::Sbfm {
                width: value_width(*ty),
                rd,
                rn,
                immr: 0,
                imms,
            });
            ctx.define(result, *ty);
        }
        Inst::Uextend { ty, value } => {
            let (rd, rn, from) = width_change(ctx, result, *value, "Uextend")?;
            out.push(match (from, *ty) {
                (Type::I8, Type::I32 | Type::I64) => ubfm_w(rd, rn, 7),
                (Type::I16, Type::I32 | Type::I64) => ubfm_w(rd, rn, 15),
                (Type::I32, Type::I64) => mov_w(rd, rn),
                _ => return Err(bad_width_change("Uextend", from, *ty)),
            });
            ctx.define(result, *ty);
        }
        Inst::Ireduce { ty, value } => {
            let (rd, rn, from) = width_change(ctx, result, *value, "Ireduce")?;
            if (from, *ty) != (Type::I64, Type::I32) {
                return Err(bad_width_change("Ireduce", from, *ty));
            }
            out.push(mov_w(rd, rn));
            ctx.define(result, *ty);
        }
        Inst::Load { ty, addr, offset } => {
            let rt = result_reg(ctx, result, "Load without result")?;
            out.push(load_inst(*ty, rt, ctx.reg(*addr)?, *offset));
            ctx.define(result, *ty);
        }
        Inst::Store {
            value,
            addr,
            offset,
        } => {
            let ty = ctx.type_of(*value)?;
            out.push(store_inst(ty, ctx.reg(*value)?, ctx.reg(*addr)?, *offset));
        }
        Inst::Call { callee, args, ret } => lower_call(callee, args, *ret, result, ctx, out)?,
    }
    Ok(())
}

// args move into x0..x7 as one parallel move, `bl` is recorded for the object writer's relocation,
// and the result comes out of x0. the result type is not in the ir, so it is a full 64-bit value
fn lower_call(
    callee: &str,
    args: &[Value],
    ret: Option<Type>,
    result: Option<Value>,
    ctx: &mut Ctx,
    out: &mut Vec<MachInst>,
) -> Result<(), LowerError> {
    if args.len() > MAX_CALL_ARGS {
        return Err(LowerError::CallArgs {
            callee: callee.into(),
            args: args.len(),
        });
    }
    let moves = args
        .iter()
        .zip(0u8..)
        .map(|(arg, number)| Ok((Reg(number), ctx.reg(*arg)?)))
        .collect::<Result<Vec<_>, LowerError>>()?;
    out.extend(
        resolve(&moves)
            .into_iter()
            .map(|(dest, source)| mov(dest, source)),
    );
    ctx.relocate(out.len(), RelocationKind::Branch26, callee);
    out.push(MachInst::Bl { offset_words: 0 });
    if let Some(value) = result {
        let rd = ctx.reg(value)?;
        if rd != RETURN_REG {
            out.push(mov(rd, RETURN_REG));
        }
        let ty = ret.ok_or(LowerError::Unsupported {
            inst: "Call result without a return type",
        })?;
        ctx.define(Some(value), ty);
    }
    Ok(())
}

// frame, outermost first: the frame record (functions that call), the callee-saved pairs, then the
// slot frame. sp stays 16-byte aligned: each pair is 16 bytes, an odd register pairs with xzr.
// entry param moves into callee-saved registers come last
struct Prologue {
    setup: Vec<MachInst>,
    epilogue: Vec<MachInst>,
}

fn pair_of(chunk: &[Reg]) -> (Reg, Reg) {
    (chunk[0], chunk.get(1).copied().unwrap_or(XZR))
}

fn build_prologue(function: &Function, ctx: &Ctx, has_call: bool) -> Result<Prologue, LowerError> {
    let (mut setup, mut tail) = (Vec::new(), Vec::new());
    if has_call {
        setup.push(MachInst::StpPre {
            rt: FRAME_POINTER,
            rt2: Reg::LR,
            rn: Reg::SP,
            offset: -16,
        });
        setup.push(MachInst::AddImm {
            width: WIDTH,
            rd: FRAME_POINTER,
            rn: Reg::SP,
            imm12: 0,
            shift12: false,
        });
        tail.push(MachInst::LdpPost {
            rt: FRAME_POINTER,
            rt2: Reg::LR,
            rn: Reg::SP,
            offset: 16,
        });
    }
    let saved = ctx.allocation.callee_saved();
    for chunk in saved.chunks(2) {
        let (rt, rt2) = pair_of(chunk);
        setup.push(MachInst::StpPre {
            rt,
            rt2,
            rn: Reg::SP,
            offset: -16,
        });
        tail.push(MachInst::LdpPost {
            rt,
            rt2,
            rn: Reg::SP,
            offset: 16,
        });
    }
    if ctx.frame.size > 0 {
        setup.extend(grow_frame(ctx.frame.size)?);
        tail.extend(release_frame(ctx.frame.size)?.into_iter().rev());
    }
    tail.reverse();
    setup.extend(arrival_moves(function, &ctx.allocation)?);
    Ok(Prologue {
        setup,
        epilogue: tail,
    })
}

// a param live across a call arrives in x0..x7 but lives in a callee-saved register
fn arrival_moves(
    function: &Function,
    allocation: &Allocation,
) -> Result<Vec<MachInst>, LowerError> {
    let params = function
        .blocks
        .first()
        .map_or(&[][..], |block| &block.params);
    let moves = params
        .iter()
        .zip(0u8..)
        .map(|((value, _), number)| Ok((allocation.reg(*value)?, Reg(number))))
        .collect::<Result<Vec<_>, LowerError>>()?;
    Ok(resolve(&moves)
        .into_iter()
        .map(|(dest, source)| mov(dest, source))
        .collect())
}

// every edge moves its args into the target's param registers as one parallel assignment: the
// resolver orders the moves and breaks cycles through x16, so a back edge that swaps its own
// header params (reading the old ones while writing the new ones) is correct
fn transfer(
    function: &Function,
    allocation: &Allocation,
    target: BlockId,
    args: &[Value],
    out: &mut Vec<MachInst>,
) -> Result<(), LowerError> {
    let params = &function.blocks[target.0 as usize].params;
    if params.len() != args.len() {
        return Err(LowerError::ArgCount {
            target,
            args: args.len(),
            params: params.len(),
        });
    }
    let moves = args
        .iter()
        .zip(params)
        .map(|(arg, (param, _))| Ok((allocation.reg(*param)?, allocation.reg(*arg)?)))
        .collect::<Result<Vec<_>, LowerError>>()?;
    out.extend(
        resolve(&moves)
            .into_iter()
            .map(|(dest, source)| mov(dest, source)),
    );
    Ok(())
}

fn lower_jump(
    function: &Function,
    allocation: &Allocation,
    target: BlockId,
    args: &[Value],
    out: &mut Vec<MachInst>,
    fixups: &mut Vec<(usize, BlockId)>,
) -> Result<(), LowerError> {
    transfer(function, allocation, target, args, out)?;
    fixups.push((out.len(), target));
    out.push(MachInst::B { offset_words: 0 });
    Ok(())
}

fn words_between(from: usize, to: usize) -> Result<i32, LowerError> {
    let delta = to as i64 - from as i64;
    i32::try_from(delta).map_err(|_| {
        EncodeError::OutOfRange {
            operand: "offset_words",
            value: delta,
            min: i64::from(i32::MIN),
            max: i64::from(i32::MAX),
        }
        .into()
    })
}

// edge splitting: `cbz cond, else`; then-edge moves; `b then`; else: else-edge moves; `b else`.
// cbz reads cond before any move and the edges never both run, so each is its own parallel move.
// the test width follows cond's type: an I32 register's upper half is garbage, so it tests w
fn lower_branch(
    function: &Function,
    ctx: &Ctx,
    cond: Value,
    edges: [(BlockId, &[Value]); 2],
    out: &mut Vec<MachInst>,
    fixups: &mut Vec<(usize, BlockId)>,
) -> Result<(), LowerError> {
    let width = value_width(ctx.type_of(cond)?);
    let rt = ctx.reg(cond)?;
    let test = out.len();
    out.push(MachInst::Cbz {
        width,
        rt,
        offset_words: 0,
    });
    let [(then_block, then_args), (else_block, else_args)] = edges;
    lower_jump(
        function,
        &ctx.allocation,
        then_block,
        then_args,
        out,
        fixups,
    )?;
    out[test] = MachInst::Cbz {
        width,
        rt,
        offset_words: words_between(test, out.len())?,
    };
    lower_jump(
        function,
        &ctx.allocation,
        else_block,
        else_args,
        out,
        fixups,
    )
}

fn lower_terminator(
    function: &Function,
    ctx: &Ctx,
    terminator: &Terminator,
    epilogue: &[MachInst],
    out: &mut Vec<MachInst>,
    fixups: &mut Vec<(usize, BlockId)>,
) -> Result<(), LowerError> {
    match terminator {
        Terminator::Return(value) => {
            if let Some(value) = value {
                let source = ctx.reg(*value)?;
                if source != RETURN_REG {
                    out.push(mov(RETURN_REG, source));
                }
            }
            out.extend_from_slice(epilogue);
            out.push(MachInst::Ret { rn: Reg::LR });
            Ok(())
        }
        Terminator::Unreachable => {
            out.push(MachInst::Brk { imm16: 0 });
            Ok(())
        }
        Terminator::Jump { target, args } => {
            lower_jump(function, &ctx.allocation, *target, args, out, fixups)
        }
        Terminator::Branch {
            cond,
            then_block,
            then_args,
            else_block,
            else_args,
        } => lower_branch(
            function,
            ctx,
            *cond,
            [
                (*then_block, then_args.as_slice()),
                (*else_block, else_args.as_slice()),
            ],
            out,
            fixups,
        ),
    }
}

/// Lower `function` to machine instructions, blocks laid out in index order
/// and every jump offset resolved after layout. The prologue (`sub sp`) sits before
/// block 0's start, so a jump back to block 0 skips it; entry params arrive in x0..x7 and are
/// precolored there by [`allocate`].
#[must_use = "the lowered instructions or the refusal must be handled"]
pub fn lower_insts(function: &Function) -> Result<Vec<MachInst>, LowerError> {
    lower_function(function).map(|lowered| lowered.insts)
}

/// [`lower_insts`] plus the `bl` sites, which the object writer turns into relocations.
///
/// A function containing a `Call` is framed: `stp x29, x30, [sp, #-16]!`, `add x29, sp, #0`, then
/// the callee-saved pairs, then the slot `sub sp`; every `ret` is preceded by the mirror image.
#[must_use = "the lowered function or the refusal must be handled"]
pub fn lower_function(function: &Function) -> Result<Lowered, LowerError> {
    verify(function)?;
    let allocation = allocate(function)?;
    let frame = layout(function)?;
    let types = function
        .blocks
        .iter()
        .flat_map(|block| block.params.iter().map(|(value, ty)| (*value, *ty)))
        .collect();
    let mut ctx = Ctx {
        allocation,
        types,
        frame,
        relocations: Vec::new(),
    };
    let has_call = function
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .any(|(_, inst)| matches!(inst, Inst::Call { .. }));
    let prologue = build_prologue(function, &ctx, has_call)?;
    let mut out = prologue.setup;
    let mut starts = Vec::with_capacity(function.blocks.len());
    let mut fixups = Vec::new();
    for block in &function.blocks {
        starts.push(out.len());
        for (result, inst) in &block.insts {
            lower_inst(inst, *result, &mut ctx, &mut out)?;
        }
        if let Some(terminator) = &block.terminator {
            lower_terminator(
                function,
                &ctx,
                terminator,
                &prologue.epilogue,
                &mut out,
                &mut fixups,
            )?;
        }
    }
    for (at, target) in fixups {
        out[at] = MachInst::B {
            offset_words: words_between(at, starts[target.0 as usize])?,
        };
    }
    Ok(Lowered {
        insts: out,
        relocations: ctx.relocations,
    })
}

/// Lower `function` to encoded little-endian instruction words.
#[must_use = "the lowered words or the refusal must be handled"]
pub fn lower(function: &Function) -> Result<Vec<u32>, LowerError> {
    lower_insts(function)?
        .into_iter()
        .map(|inst| encode(inst).map_err(LowerError::from))
        .collect()
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::string::ToString;
    use alloc::vec;

    use super::*;
    use crate::ir::Block;

    fn function_of(insts: Vec<(Option<Value>, Inst)>) -> Function {
        let block = Block {
            params: vec![],
            insts,
            terminator: Some(Terminator::Return(None)),
        };
        Function {
            name: "f".to_string(),
            params: vec![],
            returns: None,
            slots: vec![],
            blocks: vec![block],
        }
    }

    fn iconst(result: u32, ty: Type) -> (Option<Value>, Inst) {
        (Some(Value(result)), Inst::Iconst { ty, imm: 0 })
    }

    // each case: the function, and the instruction its memory access must lower to
    fn narrow_memory_cases() -> Vec<(Function, MachInst)> {
        let mut cases = Vec::new();
        for ty in [Type::I8, Type::I16] {
            let load = Inst::Load {
                ty,
                addr: Value(0),
                offset: 0,
            };
            let load_function = function_of(vec![iconst(0, Type::Ptr), (Some(Value(1)), load)]);
            let allocation = allocate(&load_function).expect("allocate load");
            let (rt, rn) = (allocation.reg(Value(1)), allocation.reg(Value(0)));
            let (rt, rn, offset) = (rt.expect("rt"), rn.expect("rn"), 0);
            let loaded = match ty {
                Type::I8 => MachInst::Ldrb { rt, rn, offset },
                _ => MachInst::Ldrh { rt, rn, offset },
            };
            cases.push((load_function, loaded));
            let store = Inst::Store {
                value: Value(1),
                addr: Value(0),
                offset: 0,
            };
            let store_function =
                function_of(vec![iconst(0, Type::Ptr), iconst(1, ty), (None, store)]);
            let allocation = allocate(&store_function).expect("allocate store");
            let (rt, rn) = (allocation.reg(Value(1)), allocation.reg(Value(0)));
            let (rt, rn, offset) = (rt.expect("rt"), rn.expect("rn"), 0);
            let stored = match ty {
                Type::I8 => MachInst::Strb { rt, rn, offset },
                _ => MachInst::Strh { rt, rn, offset },
            };
            cases.push((store_function, stored));
        }
        cases
    }

    #[test]
    fn lowers_narrow_memory_access() {
        let cases = narrow_memory_cases();
        let lowered = cases
            .iter()
            .filter(|(function, expected)| {
                let words = lower(function).expect("narrow access lowers");
                // the access is the last instruction before `ret`
                let emitted = words.len().checked_sub(2).and_then(|at| words.get(at));
                emitted == Some(&encode(*expected).expect("expected encodes"))
            })
            .count();
        std::println!("lowered={lowered} of={}", cases.len());
        assert_eq!((lowered, cases.len()), (4, 4));
    }

    fn load_at(ty: Type, offset: u32) -> Function {
        let load = Inst::Load {
            ty,
            addr: Value(0),
            offset,
        };
        function_of(vec![iconst(0, Type::Ptr), (Some(Value(1)), load)])
    }

    fn store_at(ty: Type, offset: u32) -> Function {
        let store = Inst::Store {
            value: Value(1),
            addr: Value(0),
            offset,
        };
        function_of(vec![iconst(0, Type::Ptr), iconst(1, ty), (None, store)])
    }

    fn expected_access(function: &Function, ty: Type, offset: u32, is_load: bool) -> MachInst {
        let allocation = allocate(function).expect("allocate access");
        let (rt, rn) = (allocation.reg(Value(1)), allocation.reg(Value(0)));
        let (rt, rn) = (rt.expect("rt"), rn.expect("rn"));
        let width = value_width(ty);
        match (ty, is_load) {
            (Type::I8, true) => MachInst::Ldrb { rt, rn, offset },
            (Type::I16, true) => MachInst::Ldrh { rt, rn, offset },
            (_, true) => MachInst::Ldr {
                width,
                rt,
                rn,
                offset,
            },
            (Type::I8, false) => MachInst::Strb { rt, rn, offset },
            (Type::I16, false) => MachInst::Strh { rt, rn, offset },
            (_, false) => MachInst::Str {
                width,
                rt,
                rn,
                offset,
            },
        }
    }

    #[test]
    fn lowers_offset_accesses() {
        let cases = [
            (Type::I32, 8, true),
            (Type::I64, 24, true),
            (Type::I8, 5, true),
            (Type::I16, 6, true),
            (Type::I32, 12, false),
            (Type::I64, 16, false),
            (Type::I8, 3, false),
            (Type::I16, 10, false),
        ];
        let lowered = cases
            .iter()
            .filter(|(ty, offset, is_load)| {
                let function = if *is_load {
                    load_at(*ty, *offset)
                } else {
                    store_at(*ty, *offset)
                };
                let expected = expected_access(&function, *ty, *offset, *is_load);
                let words = lower(&function).expect("offset access lowers");
                let emitted = words.len().checked_sub(2).and_then(|at| words.get(at));
                emitted == Some(&encode(expected).expect("expected encodes"))
            })
            .count();
        std::println!("lowered={lowered} of={}", cases.len());
        assert_eq!((lowered, cases.len()), (8, 8));
    }

    #[test]
    fn refuses_misaligned_offset() {
        let error = lower(&load_at(Type::I32, 6)).expect_err("offset 6 is not word aligned");
        assert!(
            matches!(
                error,
                LowerError::Encode(EncodeError::Misaligned { value: 6, .. })
            ),
            "{error:?}"
        );
    }

    #[test]
    fn refuses_out_of_range_offset() {
        let error = lower(&load_at(Type::I32, 16384)).expect_err("offset 16384 is past the form");
        assert!(
            matches!(
                error,
                LowerError::Encode(EncodeError::OutOfRange {
                    value: 16384,
                    max: 16380,
                    ..
                })
            ),
            "{error:?}"
        );
        lower(&load_at(Type::I32, 16380)).expect("offset 16380 is the last in range");
    }

    #[test]
    fn refuses_nine_call_args() {
        let mut insts: Vec<_> = (0..9).map(|id| iconst(id, Type::I32)).collect();
        insts.push((
            None,
            Inst::Call {
                callee: "sum9".to_string(),
                args: (0..9).map(Value).collect(),
                ret: None,
            },
        ));
        let error = lower(&function_of(insts)).expect_err("9 args exceed x0..x7");
        std::println!("refusal: {error}");
        assert_eq!(
            error,
            LowerError::CallArgs {
                callee: "sum9".to_string(),
                args: 9
            }
        );
    }

    fn arithmetic_case(name: &'static str, ty: Type) -> (&'static str, Type, Function) {
        let (lhs, rhs) = (Value(0), Value(1));
        let inst = match name {
            "Iadd" => Inst::Iadd(lhs, rhs),
            "Isub" => Inst::Isub(lhs, rhs),
            "Imul" => Inst::Imul(lhs, rhs),
            _ => Inst::Icmp {
                cond: IrCond::Slt,
                a: lhs,
                b: rhs,
            },
        };
        let insts = vec![iconst(0, ty), iconst(1, ty), (Some(Value(2)), inst)];
        (name, ty, function_of(insts))
    }

    #[test]
    fn refuses_narrow_arithmetic_and_compare() {
        let cases: Vec<_> = ["Iadd", "Isub", "Imul", "Icmp"]
            .into_iter()
            .flat_map(|name| {
                [Type::I8, Type::I16]
                    .into_iter()
                    .map(move |ty| arithmetic_case(name, ty))
            })
            .collect();
        let refused = cases
            .iter()
            .filter(|(name, ty, function)| {
                let error = lower(function).expect_err("narrow operands are refused");
                let named = matches!(
                    error,
                    LowerError::UnsupportedType { inst, ty: found } if inst == *name && found == *ty
                );
                named && error.to_string().contains(&std::format!("{ty:?}"))
            })
            .count();
        std::println!("refused={refused} of={}", cases.len());
        assert_eq!((refused, cases.len()), (8, 8));
    }

    fn width_change_function(kind: &str, from: Type, to: Type) -> Function {
        let value = Value(0);
        let inst = match kind {
            "Sextend" => Inst::Sextend { ty: to, value },
            "Uextend" => Inst::Uextend { ty: to, value },
            _ => Inst::Ireduce { ty: to, value },
        };
        function_of(vec![iconst(0, from), (Some(Value(1)), inst)])
    }

    fn bitfield_of(signed: bool, width: Width, rd: Reg, rn: Reg, imms: u8) -> MachInst {
        if signed {
            MachInst::Sbfm {
                width,
                rd,
                rn,
                immr: 0,
                imms,
            }
        } else {
            MachInst::Ubfm {
                width,
                rd,
                rn,
                immr: 0,
                imms,
            }
        }
    }

    fn division_words(inst: Inst, ty: Type) -> (Vec<u32>, Reg, Reg, Reg) {
        let insts = vec![iconst(0, ty), iconst(1, ty), (Some(Value(2)), inst)];
        let function = function_of(insts);
        let allocation = allocate(&function).expect("allocate");
        let reg = |id| allocation.reg(Value(id)).expect("register");
        let (rd, rn, rm) = (reg(2), reg(0), reg(1));
        (lower(&function).expect("division lowers"), rd, rn, rm)
    }

    #[test]
    fn lowers_i32_sdiv_to_the_w_form_sdiv() {
        let (words, rd, rn, rm) = division_words(Inst::Sdiv(Value(0), Value(1)), Type::I32);
        let expected = MachInst::Sdiv {
            width: Width::W32,
            rd,
            rn,
            rm,
        };
        let emitted = words.len().checked_sub(2).and_then(|at| words.get(at));
        assert_eq!(emitted, Some(&encode(expected).expect("sdiv encodes")));
    }

    #[test]
    fn lowers_i64_udiv_to_the_x_form_udiv() {
        let (words, rd, rn, rm) = division_words(Inst::Udiv(Value(0), Value(1)), Type::I64);
        let expected = MachInst::Udiv {
            width: Width::W64,
            rd,
            rn,
            rm,
        };
        let emitted = words.len().checked_sub(2).and_then(|at| words.get(at));
        assert_eq!(emitted, Some(&encode(expected).expect("udiv encodes")));
    }

    #[test]
    fn lowers_extend_and_reduce() {
        let (w, x) = (Width::W32, Width::W64);
        // kind, from, to, then the form the word must equal
        let cases = [
            ("Sextend", Type::I8, Type::I32, Some((true, w, 7))),
            ("Sextend", Type::I8, Type::I64, Some((true, x, 7))),
            ("Sextend", Type::I16, Type::I32, Some((true, w, 15))),
            ("Sextend", Type::I16, Type::I64, Some((true, x, 15))),
            ("Sextend", Type::I32, Type::I64, Some((true, x, 31))),
            ("Uextend", Type::I8, Type::I32, Some((false, w, 7))),
            ("Uextend", Type::I8, Type::I64, Some((false, w, 7))),
            ("Uextend", Type::I16, Type::I32, Some((false, w, 15))),
            ("Uextend", Type::I16, Type::I64, Some((false, w, 15))),
            ("Uextend", Type::I32, Type::I64, None),
            ("Ireduce", Type::I64, Type::I32, None),
        ];
        let lowered = cases
            .iter()
            .filter(|(kind, from, to, form)| {
                let function = width_change_function(kind, *from, *to);
                let allocation = allocate(&function).expect("allocate");
                let rd = allocation.reg(Value(1)).expect("rd");
                let rn = allocation.reg(Value(0)).expect("rn");
                let expected = match form {
                    Some((signed, width, imms)) => bitfield_of(*signed, *width, rd, rn, *imms),
                    None => mov_w(rd, rn),
                };
                let words = lower(&function).expect("valid width change lowers");
                let emitted = words.len().checked_sub(2).and_then(|at| words.get(at));
                emitted == Some(&encode(expected).expect("expected encodes"))
            })
            .count();
        std::println!("lowered={lowered} of={}", cases.len());
        assert_eq!((lowered, cases.len()), (11, 11));
    }

    #[test]
    fn refuses_bad_width_change() {
        let cases = [
            ("Sextend", Type::I64, Type::I32),
            ("Uextend", Type::I32, Type::I32),
            ("Ireduce", Type::I32, Type::I64),
            ("Sextend", Type::Ptr, Type::I64),
            ("Uextend", Type::I32, Type::Ptr),
        ];
        let refused = cases
            .iter()
            .filter(|(kind, from, to)| {
                let error = lower(&width_change_function(kind, *from, *to))
                    .expect_err("bad width change is refused");
                let text = error.to_string();
                text.contains(&std::format!("{from:?}")) && text.contains(&std::format!("{to:?}"))
            })
            .count();
        std::println!("refused={refused} of={}", cases.len());
        assert_eq!((refused, cases.len()), (5, 5));
    }

    #[test]
    fn lowers_data_addr() {
        let function = function_of(vec![(
            Some(Value(0)),
            Inst::DataAddr {
                symbol: "nova_data".to_string(),
            },
        )]);
        let lowered = lower_function(&function).expect("lower");
        let rd = match lowered.insts[0] {
            MachInst::Adrp { rd } => rd,
            other => panic!("first instruction is {other:?}, not adrp"),
        };
        let add = MachInst::AddImm {
            width: Width::W64,
            rd,
            rn: rd,
            imm12: 0,
            shift12: false,
        };
        let words = lower(&function).expect("lower words");
        assert_eq!(words[0], encode(MachInst::Adrp { rd }).expect("adrp"));
        assert_eq!(words[1], encode(add).expect("add"));
        let expected = vec![
            Relocation {
                offset: 0,
                kind: RelocationKind::Page21,
                symbol: "nova_data".to_string(),
            },
            Relocation {
                offset: 4,
                kind: RelocationKind::PageOff12,
                symbol: "nova_data".to_string(),
            },
        ];
        std::println!(
            "data_relocs={} of={}",
            lowered.relocations.len(),
            expected.len()
        );
        assert_eq!(lowered.relocations, expected);
    }

    #[test]
    fn lowers_unreachable_to_brk() {
        let block = Block {
            params: vec![],
            insts: vec![],
            terminator: Some(Terminator::Unreachable),
        };
        let function = Function {
            blocks: vec![block],
            ..function_of(vec![])
        };
        let words = lower(&function).expect("lower");
        assert_eq!(words, vec![0xd420_0000]);
    }

    fn frame_function(sizes: &[u32], insts: Vec<(Option<Value>, Inst)>) -> Function {
        let slots = sizes.iter().map(|&size| Slot { size, align: 16 }).collect();
        Function {
            slots,
            ..function_of(insts)
        }
    }

    fn words_of(insts: &[MachInst]) -> Vec<u32> {
        insts
            .iter()
            .map(|inst| encode(*inst).expect("expected instruction encodes"))
            .collect()
    }

    fn same_words(actual: &[MachInst], expected: &[MachInst]) -> usize {
        if actual.len() != expected.len() {
            return 0;
        }
        let (left, right) = (words_of(actual), words_of(expected));
        left.iter().zip(&right).filter(|(a, b)| a == b).count()
    }

    fn sub_sp(imm12: u16, shift12: bool) -> MachInst {
        sub_imm(Reg::SP, Reg::SP, imm12, shift12)
    }

    fn add_sp(imm12: u16, shift12: bool) -> MachInst {
        add_imm(Reg::SP, Reg::SP, imm12, shift12)
    }

    fn lowered_insts(function: &Function) -> Vec<MachInst> {
        lower_insts(function).expect("function lowers")
    }

    #[test]
    fn lowers_big_frame() {
        let store_through_slot = vec![
            (Some(Value(0)), Inst::SlotAddr(crate::ir::SlotId(1))),
            iconst(1, Type::I64),
            (
                None,
                Inst::Store {
                    value: Value(1),
                    addr: Value(0),
                    offset: 0,
                },
            ),
        ];
        let function = frame_function(&[32016, 16], store_through_slot);
        let insts = lowered_insts(&function);
        let growth = [
            MachInst::Movz {
                width: WIDTH,
                rd: LOOP_COUNTER,
                imm16: 7,
                hw: 0,
            },
            sub_sp(1, true),
            probe(),
            sub_imm(LOOP_COUNTER, LOOP_COUNTER, 1, false),
            MachInst::Cbnz {
                width: WIDTH,
                rt: LOOP_COUNTER,
                offset_words: -3,
            },
            sub_sp(3360, false),
            probe(),
        ];
        let rd = match insts[growth.len()] {
            MachInst::AddImm { rd, .. } => rd,
            other => panic!("first body instruction is {other:?}, not the slot add"),
        };
        let slot = [add_imm(rd, Reg::SP, 7, true), add_imm(rd, rd, 3344, false)];
        let release = [add_sp(7, true), add_sp(3360, false)];
        let end = insts.len();
        let (growth_words, release_words) = (
            same_words(&insts[..growth.len()], &growth),
            same_words(&insts[end - 3..end - 1], &release),
        );
        let slot_words = same_words(&insts[growth.len()..growth.len() + 2], &slot);
        std::println!(
            "big_frame growth_words={growth_words} of=7 release_words={release_words} of=2 slot_words={slot_words} of=2"
        );
        assert_eq!((growth_words, release_words, slot_words), (7, 2, 2));
    }

    #[test]
    fn lowers_imm12_slot_offsets() {
        let slot_ids = [
            crate::ir::SlotId(1),
            crate::ir::SlotId(2),
            crate::ir::SlotId(3),
        ];
        // pad of 4095, then byte slots at 4095, 4096 and 4097 once the pad is 4095 bytes
        let function = Function {
            slots: vec![
                Slot {
                    size: 4095,
                    align: 1,
                },
                Slot { size: 1, align: 1 },
                Slot { size: 1, align: 1 },
                Slot { size: 1, align: 1 },
            ],
            ..function_of(
                slot_ids
                    .iter()
                    .zip(0..)
                    .map(|(slot, result)| (Some(Value(result)), Inst::SlotAddr(*slot)))
                    .collect(),
            )
        };
        let insts = lowered_insts(&function);
        let growth = grow_frame(4112).expect("frame grows").len();
        let body = &insts[growth..];
        let rds = (0..3)
            .map(|index| {
                allocate(&function)
                    .expect("allocate")
                    .reg(Value(index))
                    .expect("rd")
            })
            .collect::<Vec<_>>();
        let at_4095 = [add_imm(rds[0], Reg::SP, 4095, false)];
        let at_4096 = [add_imm(rds[1], Reg::SP, 1, true)];
        let at_4097 = [
            add_imm(rds[2], Reg::SP, 1, true),
            add_imm(rds[2], rds[2], 1, false),
        ];
        let single = same_words(&body[0..1], &at_4095) + same_words(&body[1..2], &at_4096);
        let double = usize::from(same_words(&body[2..4], &at_4097) == 2);
        std::println!("offsets single={single} of=2 double={double} of=1");
        assert_eq!((single, double), (2, 1));
    }

    fn probe_count(insts: &[MachInst]) -> usize {
        insts.iter().filter(|inst| **inst == probe()).count()
    }

    fn loop_pages(insts: &[MachInst]) -> Option<u32> {
        let looped = insts.iter().any(|inst| {
            matches!(inst, MachInst::Cbnz { rt, offset_words: -3, .. } if *rt == LOOP_COUNTER)
        });
        insts.iter().find_map(|inst| match inst {
            MachInst::Movz { rd, imm16, .. } if looped && *rd == LOOP_COUNTER => {
                Some(u32::from(*imm16))
            }
            _ => None,
        })
    }

    #[test]
    fn counts_probes_like_llvm() {
        // frame, LLVM static probes, LLVM loop, probes executed
        let frames = [
            (1024u32, 0usize, false, 0u32),
            (1040, 1, false, 1),
            (4016, 1, false, 1),
            (4096, 1, false, 1),
            (4112, 1, false, 1),
            (5120, 1, false, 1),
            (5152, 2, false, 2),
            (8112, 2, false, 2),
            (12096, 3, false, 3),
            (20464, 5, false, 5),
            (20480, 1, true, 5),
            (20496, 1, true, 5),
            (24496, 2, true, 6),
            (262064, 2, true, 64),
        ];
        let (mut statics, mut loops, mut executed) = (0, 0, 0);
        for (frame, llvm_static, llvm_loop, llvm_executed) in frames {
            let insts = lowered_insts(&frame_function(&[frame], vec![]));
            let pages = loop_pages(&insts);
            let count = probe_count(&insts);
            let ran = match pages {
                Some(pages) => count - 1 + pages as usize,
                None => count,
            };
            statics += usize::from(count == llvm_static);
            loops += usize::from(pages.is_some() == llvm_loop);
            executed += usize::from(ran == llvm_executed as usize);
        }
        std::println!(
            "probes frames={} static_match={statics} of={} loop_match={loops} of={} executed_match={executed} of={}",
            frames.len(),
            frames.len(),
            frames.len(),
            frames.len()
        );
        assert_eq!((statics, loops, executed), (14, 14, 14));
    }

    fn eight_param_function(frame: u32) -> Function {
        let params = (0..8).map(|index| (Value(index), Type::I64)).collect();
        let block = Block {
            params,
            insts: vec![],
            terminator: Some(Terminator::Return(None)),
        };
        Function {
            params: vec![Type::I64; 8],
            slots: vec![Slot {
                size: frame,
                align: 16,
            }],
            blocks: vec![block],
            ..function_of(vec![])
        }
    }

    fn written(inst: &MachInst) -> Option<Reg> {
        match inst {
            MachInst::Movz { rd, .. } | MachInst::SubImm { rd, .. } => Some(*rd),
            _ => None,
        }
    }

    #[test]
    fn emits_probe_loop_form() {
        let page_words = page_loop(5);
        let remainder = [sub_sp(4016, false), probe()];
        let (mut forms, mut untouched) = (0, 0);
        for (frame, tail) in [(20480u32, &remainder[..0]), (24496, &remainder[..])] {
            let insts = lowered_insts(&eight_param_function(frame));
            let growth = page_words.len() + tail.len();
            let expected = [&page_words[..], tail].concat();
            forms += usize::from(same_words(&insts[..growth], &expected) == growth);
            let clobbers = insts[..growth].iter().filter_map(written).any(|reg| {
                reg != Reg::SP && (reg.0 <= 15 && reg != LOOP_COUNTER || (19..=28).contains(&reg.0))
            });
            untouched += usize::from(!clobbers);
        }
        std::println!("loop_form={forms} of=2 params_untouched={untouched} of=2");
        assert_eq!((forms, untouched), (2, 2));
    }

    #[test]
    fn refuses_frame_beyond_24_bits() {
        let error =
            lower(&frame_function(&[16_777_216], vec![])).expect_err("frame is past 24 bits");
        let refused = usize::from(
            error
                == LowerError::Encode(EncodeError::OutOfRange {
                    operand: "frame",
                    value: 16_777_216,
                    min: 0,
                    max: 16_777_215,
                }),
        );
        let insts = lowered_insts(&frame_function(&[16_777_200], vec![]));
        let control = [
            MachInst::Movz {
                width: WIDTH,
                rd: LOOP_COUNTER,
                imm16: 4095,
                hw: 0,
            },
            sub_sp(1, true),
            probe(),
            sub_imm(LOOP_COUNTER, LOOP_COUNTER, 1, false),
            MachInst::Cbnz {
                width: WIDTH,
                rt: LOOP_COUNTER,
                offset_words: -3,
            },
            sub_sp(4080, false),
            probe(),
        ];
        let lowered = usize::from(same_words(&insts[..control.len()], &control) == control.len());
        std::println!("refused={refused} of=1 control_lowered={lowered} of=1");
        assert_eq!((refused, lowered), (1, 1));
    }
}
