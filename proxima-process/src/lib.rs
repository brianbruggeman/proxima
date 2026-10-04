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
#[cfg(all(feature = "std", unix))]
pub mod command_pipe;
#[cfg(all(feature = "std", unix))]
pub mod dispatched;
#[cfg(all(feature = "std", unix))]
pub mod fd_pipe;
#[cfg(all(feature = "std", unix))]
pub mod fork_server;
#[cfg(feature = "std")]
pub mod host_grounds;
#[cfg(all(feature = "std", unix))]
pub mod ipc;
#[cfg(all(feature = "std", unix))]
pub mod libc_shim;
#[cfg(all(feature = "std", unix))]
pub mod pty;
#[cfg(all(feature = "std", unix))]
pub mod pty_config;
#[cfg(all(feature = "std", unix))]
pub mod pty_pipe;
#[cfg(all(feature = "std", unix))]
pub mod spawn;
#[cfg(all(feature = "std", windows))]
#[path = "spawn_windows.rs"]
pub mod spawn;

#[cfg(all(test, feature = "std", unix))]
mod tests;

pub use descriptor::{CommandDescriptor, Stdio};
pub use env::Env;

#[cfg(feature = "std")]
pub mod command;

#[cfg(feature = "std")]
pub use command::{Command, Output};
#[cfg(all(feature = "std", unix))]
pub use command_pipe::CommandPipe;
#[cfg(all(feature = "std", unix))]
pub use pty::{PtySize, current_terminal_size};
#[cfg(all(feature = "std", unix))]
pub use pty_config::{PtyConfig, PtySizeConfig};
#[cfg(all(feature = "std", unix))]
pub use pty_pipe::PtyCommandPipe;
#[cfg(feature = "std")]
pub use spawn::{Child, SpawnOptions};
