//! Path + env-var names for the co-located libc-interpose shim
//! compiled by `build.rs`.
//!
//! The shim itself is a separate `.dylib`/`.so` produced as a
//! build side-output (see `build.rs`). This module surfaces the
//! artifact path and the platform-correct preload env-var so a
//! spawner can hand them to children without hand-rolling the
//! OS-conditional logic.
//!
//! # Opt-in only
//!
//! No code anywhere loads the shim by default. A `Command` spawned
//! through proxima-process gets the shim only if the caller
//! explicitly opts in (e.g. via `CommandPipeBuilder::libc_shim`).
//! The parent process is never affected — the .dylib has no path
//! into the rlib's symbol space.

/// Absolute path to the compiled libc-interpose shim, written into
/// the build by `build.rs` (see the `PROXIMA_LIBC_SHIM_PATH`
/// cargo-rustc env var).
pub const PATH: &str = env!("PROXIMA_LIBC_SHIM_PATH");

/// Name of the platform-correct preload env var. macOS uses
/// `DYLD_INSERT_LIBRARIES`; Linux uses `LD_PRELOAD`. On a child
/// whose `Command.env` has `{PRELOAD_ENV_VAR}={PATH}`, the dynamic
/// loader links the shim into the address space before `main`.
#[cfg(target_os = "macos")]
pub const PRELOAD_ENV_VAR: &str = "DYLD_INSERT_LIBRARIES";

/// Name of the platform-correct preload env var. macOS uses
/// `DYLD_INSERT_LIBRARIES`; Linux uses `LD_PRELOAD`. On a child
/// whose `Command.env` has `{PRELOAD_ENV_VAR}={PATH}`, the dynamic
/// loader links the shim into the address space before `main`.
#[cfg(target_os = "linux")]
pub const PRELOAD_ENV_VAR: &str = "LD_PRELOAD";

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]
    use super::*;
    use std::path::Path;

    #[test]
    fn shim_artifact_exists_at_path() {
        let path = Path::new(PATH);
        assert!(
            path.is_file(),
            "expected built shim at {} (build.rs side-output)",
            PATH
        );
    }

    #[test]
    fn preload_env_var_name_matches_platform() {
        #[cfg(target_os = "macos")]
        assert_eq!(PRELOAD_ENV_VAR, "DYLD_INSERT_LIBRARIES");
        #[cfg(target_os = "linux")]
        assert_eq!(PRELOAD_ENV_VAR, "LD_PRELOAD");
    }
}
