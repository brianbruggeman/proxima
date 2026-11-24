//! The block vocabulary a network is written in: the math a model is made of,
//! as data. Every field is a math parameter (a width, a count, a flag that
//! changes the arithmetic); nothing names a model family.

mod attention;
mod block;
mod bounded;
mod decode;
mod ffn;
mod head;
mod input;
mod lower;
mod network;
mod norm;
mod stack;

pub use attention::Attention;
pub use block::Block;
pub use decode::Decode;
pub use ffn::{DenseFfn, RoutedFfn};
pub use head::Head;
pub use input::Input;
pub use network::Network;
pub use norm::{Norm, NormKind};
pub use stack::Stack;
