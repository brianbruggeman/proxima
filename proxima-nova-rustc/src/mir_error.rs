//! Why a MIR body could not be lowered to nova IR; each variant carries the offending item.

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MirError {
    #[error("unsupported type {ty}")]
    UnsupportedType { ty: String },
    #[error("layout of {ty} unavailable: {detail}")]
    Layout { ty: String, detail: String },
    #[error("size or align of {ty} does not fit u32")]
    SizeOverflow { ty: String },
    #[error("fat pointer type {ty}")]
    FatPointer { ty: String },
    #[error("local _{local} out of range")]
    LocalOutOfRange { local: usize },
    #[error("local _{local} is past the last slot id a u32 can index")]
    TooManyLocals { local: usize },
    #[error("unsupported constant kind {kind}")]
    UnsupportedConstant { kind: String },
    #[error("constant read failed: {detail}")]
    ConstantRead { detail: String },
    #[error("constant {value} does not fit i64")]
    ConstantTooWide { value: i128 },
    #[error("unsupported statement {statement}")]
    UnsupportedStatement { statement: String },
    #[error("unsupported rvalue {rvalue}")]
    UnsupportedRvalue { rvalue: String },
    #[error("unsupported binary operator {op}")]
    UnsupportedBinOp { op: String },
    #[error("unsupported operand {operand}")]
    UnsupportedOperand { operand: String },
    #[error("unsupported operand type {ty}")]
    UnsupportedOperandType { ty: String },
    #[error("unsupported cast {detail}")]
    UnsupportedCast { detail: String },
    #[error("unsupported switch {detail}")]
    UnsupportedSwitch { detail: String },
    #[error("unsupported terminator {terminator}")]
    UnsupportedTerminator { terminator: String },
    #[error("callee {callee} does not resolve to an instance")]
    UnresolvedCallee { callee: String },
    #[error("call with {count} arguments, the limit is 8")]
    TooManyCallArgs { count: usize },
    #[error("block index {block} out of range")]
    BlockOutOfRange { block: usize },
    #[error("StructArgument: struct {ty} passed by value")]
    StructArgument { ty: String },
    #[error("StructResult: struct {ty} returned by value")]
    StructResult { ty: String },
    #[error("NestedStruct: field {field} of struct {ty} is itself a struct")]
    NestedStruct { ty: String, field: String },
    #[error("struct {ty} has {fields} fields but its layout has {offsets} offsets")]
    StructShape {
        ty: String,
        fields: usize,
        offsets: usize,
    },
    #[error("NicheEncoding: enum layout stores its discriminant in a niche, not a direct tag")]
    NicheEncoding,
    #[error("enum {ty} has {variants} variants but its layout places {placed}")]
    EnumShape {
        ty: String,
        variants: usize,
        placed: usize,
    },
    #[error("SlotOffset: a pointer slot holds a nonzero offset into its target")]
    SlotOffset,
    #[error("FnPointer: a static or constant points at a function")]
    FnPointer,
    #[error("VTable: a static or constant points at a vtable")]
    VTable,
    #[error("TypeId: a static or constant holds a type id")]
    TypeId,
    #[error("TrackCaller: #[track_caller] body {item} needs its implicit caller location")]
    TrackCaller { item: String },
    #[error("UnwindCleanup: unwind action is not Unreachable in {terminator}")]
    UnwindCleanup { terminator: String },
    #[error("no MIR body")]
    NoBody,
}
