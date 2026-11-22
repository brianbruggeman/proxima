use alloc::vec::Vec;

use super::{Inst, Terminator, Type, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub params: Vec<(Value, Type)>,
    pub insts: Vec<(Option<Value>, Inst)>,
    pub terminator: Option<Terminator>,
}
