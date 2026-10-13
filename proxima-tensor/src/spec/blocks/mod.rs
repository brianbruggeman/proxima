//! The block vocabulary a network is written in: the math a model is made of,
//! as data. Every field is a math parameter (a width, a count, a flag that
//! changes the arithmetic); nothing names a model family.

mod block;
mod decode;
mod input;
mod network;
mod norm;
mod stack;

pub use block::Block;
pub use decode::Decode;
pub use input::Input;
pub use network::Network;
pub use norm::{Norm, NormKind};
pub use stack::Stack;
