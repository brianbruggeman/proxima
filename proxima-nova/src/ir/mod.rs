mod block;
mod dominators;
mod function;
mod inst;
mod terminator;
mod types;
mod value;
mod verify;

pub use block::Block;
pub use function::{Function, Slot};
pub use inst::{Cond, Inst};
pub use terminator::Terminator;
pub use types::Type;
pub use value::{BlockId, SlotId, Value};
pub use verify::{VerifyError, verify};
