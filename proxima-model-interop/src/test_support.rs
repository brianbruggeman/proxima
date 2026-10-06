//! Test-only helpers shared across this crate's own `#[cfg(test)]` modules
//! (`bind.rs`'s `real_openchat_file` and `quality.rs`'s own
//! `real_openchat_file`) -- kept as ONE definition here rather than a
//! private copy per module, per guiding-principle 1 (reuse first).

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::value::{MetadataArray, MetadataValue};
use proxima_gguf::{GgufModel, parse_complete, write_complete};

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
/// unset or `"concurrent"` selects [`omega::DispatchType::Concurrent`].
/// `ServingConfig::default`'s own `dispatch_type` flipped to `Serial` at
/// 3afb3db37 (a recurrent-routed residency-boundary side effect, not a per-model
/// measurement); this function's own unset-is-`Concurrent` default predates
/// that flip and was not updated to match, so a caller of this function
/// still gets `Concurrent` unless it opts into `serial` explicitly, and any
/// other value panics naming what it saw rather than silently falling back
/// to a mode the caller did not ask for.
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
/// `PROXIMA_GEMMA4_E2B_GGUF` read the way [`openchat_gguf_path`] reads
/// `PROXIMA_OPENCHAT_GGUF`: unset keeps every caller pointed at the ollama blob
/// for the E2B checkpoint on this host.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn gemma4_e2b_gguf_path() -> String {
    std::env::var("PROXIMA_GEMMA4_E2B_GGUF").unwrap_or_else(|_| {
        String::from(
            "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd",
        )
    })
}

#[cfg(unix)]
pub(crate) fn openchat_gguf_path() -> String {
    std::env::var("PROXIMA_OPENCHAT_GGUF").unwrap_or_else(|_| {
        crate::serving::ServingConfig::default()
            .model_path
            .to_string()
    })
}

/// `PROXIMA_QWEN3_GGUF` read the same way [`openchat_gguf_path`] reads
/// `PROXIMA_OPENCHAT_GGUF`: unset keeps every caller pointed at this
/// host-local Qwen3-1.7B `Q4_K_M` checkpoint (dense `qwen3` architecture,
/// `attn_q_norm`/`attn_k_norm` present at every layer, so it routes through
/// `RopePairing::SplitHalf` -- `proxima-tensor/src/spec.rs:660`). No
/// `ServingConfig` default exists for a second model family the way
/// openchat has one, so this constant path is the closest analog.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn qwen3_gguf_path() -> String {
    std::env::var("PROXIMA_QWEN3_GGUF").unwrap_or_else(|_| {
        "/Users/brianbruggeman/.ollama/models/blobs/\
         sha256-3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6"
            .to_string()
    })
}

/// `PROXIMA_QWEN3MOE_GGUF` read the same way [`qwen3_gguf_path`] reads
/// `PROXIMA_QWEN3_GGUF`: unset keeps `real_qwen3moe_file` pointed at this
/// host-local Qwen3-30B-A3B `Q4_K`/`Q6_K` checkpoint -- the real 30B-A3B
/// mixture-of-experts checkpoint `InteropError::MoeExpertShapeMismatch` was
/// first root-caused against (`qwen3moe.feed_forward_length=6144` vs.
/// `qwen3moe.expert_feed_forward_length=768`; `blk.0.ffn_gate_exps.weight`
/// GGUF `dims=[2048, 768, 128]`).
#[cfg(unix)]
pub(crate) fn qwen3moe_30b_gguf_path() -> String {
    std::env::var("PROXIMA_QWEN3MOE_GGUF").unwrap_or_else(|_| {
        "/Users/brianbruggeman/.ollama/models/blobs/\
         sha256-58574f2e94b99fb9e4391408b57e5aeaaaec10f6384e9a699fc2cb43a5c8eabf"
            .to_string()
    })
}

/// `PROXIMA_QWEN35MOE_GGUF` read the same way [`qwen3moe_30b_gguf_path`]
/// reads `PROXIMA_QWEN3MOE_GGUF`: unset keeps a recurrent-routed fixture test
/// pointed at this host-local `qwen3.6:35b-a3b` (`general.architecture =
/// recurrent-routed`) checkpoint -- the one `real_qwen35moe_registry_probe.rs`'s
/// own doc already names as the real blob every recurrent-routed-specific
/// diagnostic in this crate resolves against.
#[cfg(all(test, feature = "metal-output-placement", target_os = "macos"))]
pub(crate) fn qwen35moe_gguf_path() -> String {
    std::env::var("PROXIMA_QWEN35MOE_GGUF").unwrap_or_else(|_| {
        "/Users/brianbruggeman/.ollama/models/blobs/\
         sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d"
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
        None => panic!(
            "no host-local gguf fixture at {path}: stage one at this hardcoded path (no environment override exists)"
        ),
    }
}

/// `e2b-it-qat`'s KV-relevant header (`ollama /api/show`, 2026-09-29):
/// 35 blocks, one kv head everywhere, every fifth layer full attention (key
/// length 512) and the rest sliding (key length 256, window 512), the last 20
/// layers sharing an earlier layer's KV.
pub(crate) fn gemma4_e2b_header() -> ParsedGguf {
    parsed_header(vec![
        (
            "general.architecture",
            MetadataValue::String("gemma4".to_string()),
        ),
        ("gemma4.block_count", MetadataValue::U32(35)),
        (
            "gemma4.attention.head_count_kv",
            MetadataValue::Array(MetadataArray::U32(vec![1; 35])),
        ),
        (
            "gemma4.attention.sliding_window_pattern",
            MetadataValue::Array(MetadataArray::Bool(
                (0..35u32).map(|index| (index + 1) % 5 != 0).collect(),
            )),
        ),
        ("gemma4.attention.shared_kv_layers", MetadataValue::U32(20)),
        ("gemma4.attention.key_length", MetadataValue::U32(512)),
        ("gemma4.attention.key_length_swa", MetadataValue::U32(256)),
        ("gemma4.attention.sliding_window", MetadataValue::U32(512)),
    ])
}

/// A header-only GGUF (no tensors) whose metadata is exactly `metadata`,
/// written by the real encoder and parsed back by the real decoder -- the
/// bytes a checkpoint's own header would hand `crate::lowering::kv_layers` and
/// friends. The buffer is leaked because the parsed view borrows it; a test
/// fixture, not a hot path.
pub(crate) fn parsed_header(metadata: Vec<(&str, MetadataValue)>) -> ParsedGguf {
    let model = GgufModel {
        version: 3,
        metadata: metadata
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
        tensors: Vec::new(),
    };
    let bytes = match write_complete(&model) {
        Ok(bytes) => bytes,
        Err(error) => panic!("a tensor-less gguf model must encode: {error:?}"),
    };
    let leaked: &'static [u8] = Vec::leak(bytes);
    match parse_complete(leaked) {
        Ok(parsed) => parsed,
        Err(error) => panic!("bytes the gguf encoder just wrote must parse: {error:?}"),
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
