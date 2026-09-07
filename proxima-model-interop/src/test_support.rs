//! Test-only helpers shared across this crate's own `#[cfg(test)]` modules
//! (`bind.rs`'s `real_openchat_file` and `quality.rs`'s own
//! `real_openchat_file`) -- kept as ONE definition here rather than a
//! private copy per module, per guiding-principle 1 (reuse first).

/// `PROXIMA_MATH_MODE` read the same way every real-checkpoint decode test
/// in this crate reads it: `"safe"` selects [`omega::MathMode::Safe`],
/// `"fast"` selects [`omega::MathMode::Fast`], unset or `"relaxed"` selects
/// [`omega::MathMode::Relaxed`] (`ServingConfig::default`'s own `Relaxed`
/// remains the default so a test that never sets this env var keeps
/// behaving exactly as it did before this knob existed), and any other
/// value panics naming what it saw rather than silently falling back to a
/// mode the caller did not ask for.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn math_mode_from_env() -> omega::MathMode {
    match std::env::var("PROXIMA_MATH_MODE").as_deref() {
        Ok("safe") => omega::MathMode::Safe,
        Ok("fast") => omega::MathMode::Fast,
        Ok("relaxed") | Err(_) => omega::MathMode::Relaxed,
        Ok(other) => panic!("PROXIMA_MATH_MODE={other}: expected `safe`, `relaxed`, or `fast`"),
    }
}

/// `PROXIMA_DISPATCH` read the same way [`math_mode_from_env`] reads
/// `PROXIMA_MATH_MODE`: `"serial"` selects [`omega::DispatchType::Serial`],
/// unset or `"concurrent"` selects [`omega::DispatchType::Concurrent`]
/// (`ServingConfig::default`'s own `Concurrent` remains the default so a
/// test that never sets this env var keeps behaving exactly as it did before
/// this knob existed), and any other value panics naming what it saw rather
/// than silently falling back to a mode the caller did not ask for.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn dispatch_type_from_env() -> omega::DispatchType {
    match std::env::var("PROXIMA_DISPATCH").as_deref() {
        Ok("serial") => omega::DispatchType::Serial,
        Ok("concurrent") | Err(_) => omega::DispatchType::Concurrent,
        Ok(other) => panic!("PROXIMA_DISPATCH={other}: expected `serial` or `concurrent`"),
    }
}

/// `PROXIMA_OPENCHAT_GGUF` read the same way `omega/tests/device_streaming_ceiling.rs`
/// already reads it: unset keeps every caller pointed at
/// [`crate::serving::ServingConfig::default`]'s own `model_path`
/// (`serving.rs`'s `DEFAULT_MODEL_PATH`) exactly as before this knob
/// existed, set swaps in a different host-local gguf checkout (a
/// different quantization variant, most often) without touching
/// `ServingConfig` or any non-test source.
pub(crate) fn openchat_gguf_path() -> String {
    std::env::var("PROXIMA_OPENCHAT_GGUF")
        .unwrap_or_else(|_| crate::serving::ServingConfig::default().model_path.to_string())
}

/// `PROXIMA_QWEN3_GGUF` read the same way [`openchat_gguf_path`] reads
/// `PROXIMA_OPENCHAT_GGUF`: unset keeps every caller pointed at this
/// host-local Qwen3-1.7B `Q4_K_M` checkpoint (dense `qwen3` architecture,
/// `attn_q_norm`/`attn_k_norm` present at every layer, so it routes through
/// `RopePairing::SplitHalf` -- `proxima-tensor/src/spec.rs:660`). No
/// `ServingConfig` default exists for a second model family the way
/// openchat has one, so this constant path is the closest analog.
#[cfg(feature = "metal")]
pub(crate) fn qwen3_gguf_path() -> String {
    std::env::var("PROXIMA_QWEN3_GGUF").unwrap_or_else(|_| {
        "/Users/brianbruggeman/.ollama/models/blobs/\
         sha256-3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6"
            .to_string()
    })
}

/// Fails the calling `#[ignore]`d fixture test loudly, naming `env_var`
/// (when the caller resolves `path` through one) and `path` itself, when
/// the host-local checkpoint the test needs is absent -- an `#[ignore]`d
/// test only runs when a caller explicitly asks for it (`cargo test --
/// --ignored`), so silently returning success having executed nothing is
/// indistinguishable, from the caller's own exit code, from having proved
/// the thing the test's name claims. `env_var` is [`None`] for the handful
/// of fixtures (`real_mixtral_file`, `real_lfm2_hybrid_file`) that resolve
/// a hardcoded path with no environment override.
pub(crate) fn require_fixture(path: &str, env_var: Option<&str>) {
    if std::path::Path::new(path).exists() {
        return;
    }
    match env_var {
        Some(name) => panic!(
            "no host-local gguf fixture at {path}: set {name} to a valid checkpoint path, or stage one at this default path"
        ),
        None => panic!("no host-local gguf fixture at {path}: stage one at this hardcoded path (no environment override exists)"),
    }
}

#[cfg(test)]
mod tests {
    use super::require_fixture;

    /// The one shape this module gates on: a missing checkpoint must fail
    /// loudly, not return quietly having done nothing (an `#[ignore]`d
    /// fixture test that returns `Ok` on an absent checkpoint is
    /// indistinguishable, from its exit code, from one that actually ran).
    /// This proves the failure, never that some real fixture exists on the
    /// running host -- a passing fixture-present path is exactly what the
    /// `#[ignore]`d tests above already exercise on a host that has one.
    #[test]
    #[should_panic(expected = "no host-local gguf fixture at")]
    fn require_fixture_panics_when_the_path_is_absent() {
        require_fixture(
            "/nonexistent/path/held/out/for/this/test/proxima-model-interop.gguf",
            Some("PROXIMA_QWEN3_GGUF"),
        );
    }

    /// Same failure shape with no environment override to name -- the
    /// `real_mixtral_file`/`real_lfm2_hybrid_file` fixtures resolve a
    /// hardcoded path, so their panic message names only the path.
    #[test]
    #[should_panic(expected = "no environment override exists")]
    fn require_fixture_panics_naming_no_override_when_env_var_is_none() {
        require_fixture(
            "/nonexistent/path/held/out/for/this/test/proxima-model-interop.gguf",
            None,
        );
    }
}
