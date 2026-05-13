//! compile-time sizing consts emitted by build.rs from proxima-runtime.toml.

include!(concat!(env!("OUT_DIR"), "/proxima_runtime_sized.rs"));
