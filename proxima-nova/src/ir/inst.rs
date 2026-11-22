use alloc::string::String;
use alloc::vec::Vec;

use super::{SlotId, Type, Value};

/// Integer comparison condition: `S*` compare as signed, `U*` as unsigned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cond {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ult,
    Ule,
    Ugt,
    Uge,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inst {
    Iconst {
        ty: Type,
        imm: i64,
    },
    Iadd(Value, Value),
    Isub(Value, Value),
    Imul(Value, Value),
    /// signed division; aarch64 `sdiv` yields 0 for a zero divisor, so guards belong to the caller
    Sdiv(Value, Value),
    /// unsigned division; aarch64 `udiv` yields 0 for a zero divisor, so guards belong to the caller
    Udiv(Value, Value),
    /// result is an `I8` holding 0 or 1
    Icmp {
        cond: Cond,
        a: Value,
        b: Value,
    },
    /// widen `value` to the strictly wider `ty` by its sign bit
    Sextend {
        ty: Type,
        value: Value,
    },
    /// widen `value` to the strictly wider `ty` with zeros
    Uextend {
        ty: Type,
        value: Value,
    },
    /// narrow an `I64` `value` to `ty`, `I32`
    Ireduce {
        ty: Type,
        value: Value,
    },
    /// the address of a frame slot, rematerialized where it stands; yields a `Ptr`
    SlotAddr(SlotId),
    /// the address of the data symbol `symbol`, resolved by the linker; yields a `Ptr`
    DataAddr {
        symbol: String,
    },
    /// `offset` is a byte offset from `addr`
    Load {
        ty: Type,
        addr: Value,
        offset: u32,
    },
    /// `offset` is a byte offset from `addr`
    Store {
        value: Value,
        addr: Value,
        offset: u32,
    },
    /// call the function named `callee` (a symbol); the slot's result, when present, is its x0 and
    /// has type `ret`, the callee's declared return type
    Call {
        callee: String,
        args: Vec<Value>,
        ret: Option<Type>,
    },
}
