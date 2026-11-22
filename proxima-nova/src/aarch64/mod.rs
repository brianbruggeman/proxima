//! aarch64 machine-instruction payloads, their encoder, and IR lowering.
//!
//! Data only: [`MachInst`] is a plain enum, [`encode::encode`] a pure function,
//! [`lower::lower`] turns a nova function into words.
//! [`crate::pipeline`] composes them into pipes.
pub mod encode;
mod inst;
pub mod lower;
pub mod parallel_move;
pub mod regalloc;

pub use encode::{EncodeError, encode};
pub use inst::{Cond, MachInst, Reg, Width};
pub use lower::{LowerError, Lowered, lower, lower_function, lower_insts};
pub use regalloc::{Allocation, RegAllocError, allocate, live_ranges};
