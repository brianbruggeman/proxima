//! no_std + alloc layer. files in this module use `core::*` and `alloc::*` only,
//! never `std::*`.

pub mod sized;

#[cfg(feature = "runtime-prime-thread-identity")]
pub mod thread_identity;

#[cfg(any(
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-inbox-const",
    feature = "runtime-prime-inbox-dynamic",
))]
pub mod inbox;

#[cfg(feature = "runtime-prime-inbox-dynamic")]
pub use inbox::inbox_dynamic;

#[cfg(feature = "runtime-prime-timer")]
pub mod timer;

#[cfg(feature = "runtime-prime-executor")]
pub mod local_executor;

#[cfg(feature = "runtime-prime-executor")]
pub mod inline_task;
