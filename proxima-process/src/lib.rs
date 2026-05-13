#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod capabilities;
pub mod descriptor;
pub mod env;
pub mod framing;
pub mod grounds;
pub mod markers;
pub mod operators;
pub mod path;
pub mod protocol;
pub mod taint;

#[cfg(feature = "std")]
pub mod command_config;
#[cfg(feature = "std")]
pub mod command_pipe;
#[cfg(feature = "std")]
pub mod dispatched;
#[cfg(feature = "std")]
pub mod fd_pipe;
#[cfg(feature = "std")]
pub mod fork_server;
#[cfg(feature = "std")]
pub mod host_grounds;
#[cfg(feature = "std")]
pub mod ipc;
#[cfg(feature = "std")]
pub mod libc_shim;
#[cfg(feature = "std")]
pub mod pty;
#[cfg(feature = "std")]
pub mod pty_config;
#[cfg(feature = "std")]
pub mod pty_pipe;
#[cfg(feature = "std")]
pub mod spawn;

#[cfg(all(test, feature = "std"))]
mod tests;

pub use descriptor::{CommandDescriptor, Stdio};
pub use env::Env;

#[cfg(feature = "std")]
pub mod command;

#[cfg(feature = "std")]
pub use command::{Command, Output};
#[cfg(feature = "std")]
pub use command_pipe::CommandPipe;
#[cfg(feature = "std")]
pub use pty::{PtySize, current_terminal_size};
#[cfg(feature = "std")]
pub use pty_config::{PtyConfig, PtySizeConfig};
#[cfg(feature = "std")]
pub use pty_pipe::PtyCommandPipe;
#[cfg(feature = "std")]
pub use spawn::SpawnOptions;
