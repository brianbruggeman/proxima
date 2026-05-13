//! The porting workflow, mechanically: point `PROXIMA_PROFILE` at a file
//! under `<workspace>/profiles/<name>.toml` and this `build.rs` bakes that
//! profile's axes into `pub const`s. No source in this crate changes
//! between profiles — only the resolved constants do. This is the same
//! `proxima_build::resolve_profile()` call that `prime/build.rs` and
//! `proxima-time/build.rs` make for real; this crate exists to show it in
//! isolation.
//!
//! `PROXIMA_PROFILE` unset falls back to `Profile::default()` so a plain
//! `cargo build` still compiles — matching the convention documented in
//! `.github/workflows/no-std.yml`.

use std::env;
use std::path::PathBuf;

use proxima_build::{Profile, Resolved};

// build scripts fail fast on misconfiguration; a bad PROXIMA_PROFILE should
// abort the build loudly, matching prime/build.rs and proxima-time/build.rs.
#[allow(clippy::expect_used)]
fn main() {
    let resolved = if env::var("PROXIMA_PROFILE").is_ok() {
        proxima_build::resolve_profile().expect("resolve PROXIMA_PROFILE")
    } else {
        Resolved {
            profile: Profile::default(),
            profile_file: PathBuf::from("<PROXIMA_PROFILE unset — Profile::default()>"),
            env_vars: Vec::new(),
        }
    };

    proxima_build::emit_generated_module(&resolved).expect("emit proxima_profile.rs");
    proxima_build::emit_cfg_directives(&resolved);
    proxima_build::emit_rerun_directives(&resolved);
    println!("cargo:rerun-if-env-changed=PROXIMA_PROFILE");
}
