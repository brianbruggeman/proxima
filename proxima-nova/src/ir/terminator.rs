use alloc::vec::Vec;

use super::{BlockId, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    Return(Option<Value>),
    Jump {
        target: BlockId,
        args: Vec<Value>,
    },
    Branch {
        cond: Value,
        then_block: BlockId,
        then_args: Vec<Value>,
        else_block: BlockId,
        else_args: Vec<Value>,
    },
    /// control never reaches here; aarch64 lowers it to a trap
    Unreachable,
}
