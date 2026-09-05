//! Test-only helpers shared across this crate's own `#[cfg(test)]` modules
//! (`bind.rs`'s `real_openchat_file` and `quality.rs`'s own
//! `real_openchat_file`) -- kept as ONE definition here rather than a
//! private copy per module, per guiding-principle 1 (reuse first).

/// `PROXIMA_MATH_MODE` read the same way every real-checkpoint decode test
/// in this crate reads it: `"safe"` selects [`omega::MathMode::Safe`],
/// unset or `"relaxed"` selects [`omega::MathMode::Relaxed`]
/// (`ServingConfig::default`'s own `Relaxed` remains the default so a test
/// that never sets this env var keeps behaving exactly as it did before
/// this knob existed), and any other value panics naming what it saw
/// rather than silently falling back to a mode the caller did not ask for.
pub(crate) fn math_mode_from_env() -> omega::MathMode {
    match std::env::var("PROXIMA_MATH_MODE").as_deref() {
        Ok("safe") => omega::MathMode::Safe,
        Ok("relaxed") | Err(_) => omega::MathMode::Relaxed,
        Ok(other) => panic!("PROXIMA_MATH_MODE={other}: expected `safe` or `relaxed`"),
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
