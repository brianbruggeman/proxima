use alloc::string::String;
use alloc::vec::Vec;

use std::collections::BTreeMap;

use cudarc::nvrtc::CompileOptions;

use crate::cuda_kernel_cache::CacheKey;

use super::device::compile_options;

/// Options every kernel compiles with: ROCm 6.2 hides the `__shfl_*_sync` builtins the emitted text
/// calls unless this is defined; it is harmless on 6.3 and later.
pub(super) const HIPRTC_OPTIONS: &[&str] = &["-DHIP_ENABLE_WARP_SYNC_BUILTINS"];

/// The kernel disk store's name for the code object `source` compiles to on `arch` under `options`.
/// The resolved hipRTC flags carry the `--offload-arch`, so another arch or option set is another
/// key; `compiler` is the hipRTC `(major, minor)` in the key's compiler-version field, so a code
/// object is reused only by the compiler version that built it. The schema version and framing
/// are the CUDA path's ([`CacheKey`]).
pub(super) fn code_object_key(
    source: &str,
    arch: &str,
    compiler: (i32, i32),
    options: &[&str],
) -> CacheKey {
    let flags = CompileOptions {
        options: compile_options(Some(arch), options),
        ..CompileOptions::default()
    };
    CacheKey::new(source, None, compiler, &flags, (0, 0))
}

/// Built functions by entry name, each bucket holding `(source, function)` pairs.
pub(super) type FunctionCache<Function> = BTreeMap<String, Vec<(String, Function)>>;

/// The function built from exactly `source` under `entry`, running `build` only on a miss. Mirrors
/// the CUDA runtime's `cached_function`: a bucket per entry, matched on source text.
pub(super) fn cached_by_source<Function: Clone, Failure>(
    cache: &proxima_primitives::sync::blocking::Mutex<FunctionCache<Function>>,
    entry: &str,
    source: &str,
    build: impl FnOnce() -> Result<Function, Failure>,
) -> Result<Function, Failure> {
    let found = cache
        .lock()
        .get(entry)
        .and_then(|bucket| bucket.iter().find(|(built_from, _)| built_from == source))
        .map(|(_, function)| function.clone());
    if let Some(function) = found {
        return Ok(function);
    }
    let function = build()?;
    cache
        .lock()
        .entry(entry.to_owned())
        .or_default()
        .push((source.to_owned(), function.clone()));
    Ok(function)
}
