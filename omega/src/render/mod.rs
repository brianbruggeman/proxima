//! Backend-neutral rendering of numeric layouts.
//!
//! The integer steps that decode and encode a [`proxima_numeric::float_layout::FloatLayout`] are
//! one algorithm; a [`Dialect`] spells them for a target language and a [`HardwareScalar`] names
//! the wide scalar the steps compute in. `msl::dialect::Msl` is the first dialect.
//!
//! Teaching pointer: `proxima_numeric::float_layout_codec::{decode, encode}` is the CPU algorithm
//! these functions render step for step. `docs/omega/08-the-ir.md` builds the IR a backend lowers
//! and says which of that lowering is already shared here and which is still per backend.

mod dialect;
mod hardware;
mod layout;
mod refusal;

pub use dialect::{Dialect, Intrinsic};
#[cfg(test)]
pub(crate) use hardware::HARDWARE;
pub(crate) use hardware::HARDWARE_COUNT;
pub use hardware::HardwareScalar;
pub use layout::{load, store, store_nan_branch, unpack};
pub use refusal::Refusal;
