use alloc::string::String;
use alloc::vec::Vec;

use super::{Block, Type};

/// a frame slot: `size` bytes at `align`; its address is taken with `Inst::SlotAddr`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    pub size: u32,
    pub align: u32,
}

/// a block's id is its index in `blocks`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub params: Vec<Type>,
    pub returns: Option<Type>,
    pub slots: Vec<Slot>,
    pub blocks: Vec<Block>,
}
