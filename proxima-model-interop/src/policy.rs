//! Model-level runtime policy -- the knobs a llama.cpp invocation spends on
//! shaving a model into a memory budget, expressed for this workspace.
//!
//! Composition (guiding-principle 2): the same two primitives
//! `proxima_tensor::policy` uses, and no third mechanism --
//! `conflaguration::ConfigBuilder` for the layered load (compiled value, then
//! TOML, then environment, then validation) and `bon::Builder` for the fluent
//! half. Resolution is memoized in a `OnceLock` by [`active`], so nothing
//! here is read more than once per process.
//!
//! # Why this is a second policy type and not a bigger `ExecutionPolicy`
//!
//! `proxima_tensor::policy::ExecutionPolicy` holds knobs `proxima-tensor`
//! itself reads: chunk counts, worker counts, spin budgets. It has no model,
//! no KV cache, no tokenizer and no generation loop, so a `context_length` or
//! a `kv_cache_key_dtype` there would be a field its own crate could never
//! consult -- a knob whose owner cannot read it. Everything below belongs to
//! whoever binds a GGUF and drives a forward, which is this crate. The two
//! layer, they do not merge: a deployment writes one TOML with a `[tensor]`
//! section and a `[model]` section, and each crate loads its own.
//!
//! # What is wired and what is a map of the gap
//!
//! [`ModelPolicy::prefault`] is wired: it replaces the ad-hoc
//! `std::env::var("PROXIMA_PREFAULT")` that used to sit in `bind`'s
//! forward-pass harness. Every other field is a knob with nothing behind it,
//! and each one's accessor is a [`todo!`] that names exactly what is missing
//! and where the work attaches. They are present rather than omitted on
//! purpose: an absent field hides the gap in tribal knowledge, a `todo!` with
//! a sentence makes it compile-time greppable. None of them is reachable by
//! default -- every one defaults to `None`, and only a caller who explicitly
//! sets one reaches its `todo!`.

use std::sync::OnceLock;

#[cfg(feature = "config")]
use bon::Builder;
#[cfg(feature = "config")]
use conflaguration::{ConfigBuilder, Settings, Validate};
#[cfg(feature = "config")]
use serde::{Deserialize, Serialize};

/// KV-cache element type -- llama.cpp's `-ctk` / `-ctv` vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "config", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
#[non_exhaustive]
pub enum CacheDType {
    /// full-width cache, what an unquantized implementation would hold.
    #[default]
    F16,
    /// llama.cpp `q8_0` -- 8-bit blocks, the setting in the 27B-on-16-GB
    /// invocation this surface is measured against.
    Q8_0,
    /// llama.cpp `q4_0`.
    Q4_0,
}

/// Every model-level runtime knob. Defaults are [`Self::COMPILED`]: prefault
/// off, every unimplemented knob unset, which is exactly today's behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(Builder, Serialize, Deserialize, Settings))]
#[cfg_attr(feature = "config", settings(prefix = "PROXIMA_MODEL"))]
#[cfg_attr(feature = "config", serde(default))]
#[cfg_attr(feature = "config", builder(derive(Clone, Debug)))]
#[non_exhaustive]
pub struct ModelPolicy {
    /// Warm every page of the mapped GGUF through the background pool before
    /// the first forward, instead of taking minor faults inside it. The one
    /// wired knob here, and the replacement for the `PROXIMA_PREFAULT`
    /// environment read that used to live in `bind`'s harness: the key is now
    /// `PROXIMA_MODEL_PREFAULT`, resolved through the same layered chain as
    /// everything else. Off by default, because a caller serving many small
    /// models should not pay to warm a mapping it will read a slice of.
    #[cfg_attr(feature = "config", setting(default))]
    #[cfg_attr(feature = "config", builder(default = false))]
    pub prefault: bool,

    /// llama.cpp `-c`. `#[setting(skip)]`, like every other unimplemented
    /// knob here: an environment key for a knob with nothing behind it would
    /// be a lie, and a stray variable in a shell would turn into a panic. A
    /// TOML file can still name it, which is where a map of the gap belongs.
    #[cfg_attr(feature = "config", setting(skip))]
    pub context_length: Option<u32>,

    /// llama.cpp `-b`.
    #[cfg_attr(feature = "config", setting(skip))]
    pub batch_tokens: Option<u32>,

    /// llama.cpp `-ub`.
    #[cfg_attr(feature = "config", setting(skip))]
    pub micro_batch_tokens: Option<u32>,

    /// llama.cpp `-fa`.
    #[cfg_attr(feature = "config", setting(skip))]
    pub flash_attention: Option<bool>,

    /// llama.cpp `-ctk`.
    #[cfg_attr(feature = "config", setting(skip))]
    pub kv_cache_key_dtype: Option<CacheDType>,

    /// llama.cpp `-ctv`.
    #[cfg_attr(feature = "config", setting(skip))]
    pub kv_cache_value_dtype: Option<CacheDType>,

    /// llama.cpp `-ngl` (`None` = all layers on the CPU, today's only
    /// behaviour).
    #[cfg_attr(feature = "config", setting(skip))]
    pub gpu_layers: Option<u32>,

    /// llama.cpp `--no-kv-offload` inverted: `Some(false)` keeps the KV cache
    /// off the accelerator.
    #[cfg_attr(feature = "config", setting(skip))]
    pub kv_offload: Option<bool>,

    /// llama.cpp `--reasoning-budget`.
    #[cfg_attr(feature = "config", setting(skip))]
    pub reasoning_budget_tokens: Option<u32>,
}

impl Default for ModelPolicy {
    fn default() -> Self {
        Self::COMPILED
    }
}

impl ModelPolicy {
    /// Today's behaviour, field for field. The base layer of every load.
    pub const COMPILED: Self = Self {
        prefault: false,
        context_length: None,
        batch_tokens: None,
        micro_batch_tokens: None,
        flash_attention: None,
        kv_cache_key_dtype: None,
        kv_cache_value_dtype: None,
        gpu_layers: None,
        kv_offload: None,
        reasoning_budget_tokens: None,
    };

    /// Tokens of context the forward pass may address.
    pub fn context_window(&self) -> Option<u32> {
        match self.context_length {
            None => None,
            Some(_) => todo!(
                "context length (llama.cpp -c): there is no context window to size. \
                 bind::gguf_tensor_as_f32 + cpu::evaluate_parallel run exactly ONE \
                 forward at one position; sequence length enters as the symbolic \
                 extent `?0` on a per-call basis (proxima-tensor/src/spec.rs's \
                 ExtentSpec::Symbolic) and nothing retains state across calls. This \
                 knob attaches once a session type owns the `?0` binding plus the KV \
                 cache below, at which point -c is that session's allocation bound."
            ),
        }
    }

    /// Prompt batch and micro-batch, in tokens.
    pub fn batch_shape(&self) -> Option<(u32, u32)> {
        match (self.batch_tokens, self.micro_batch_tokens) {
            (None, None) => None,
            _ => todo!(
                "batch / micro-batch (llama.cpp -b / -ub): there is no batching concept. \
                 every matmul this crate drives is batch-1 -- cpu::quantized_matmul_workers \
                 chunks WEIGHT ROWS precisely because the activation side is one vector \
                 (see its doc). -b/-ub become real once a bound program carries a token \
                 axis, which is the same prerequisite as the KV cache: the batch dim is \
                 what makes prefill distinct from decode, and today we only have decode."
            ),
        }
    }

    /// Attention kernel selection.
    pub fn flash_attention_enabled(&self) -> Option<bool> {
        match self.flash_attention {
            None => None,
            Some(_) => todo!(
                "flash attention (llama.cpp -fa): not implemented. attention here is \
                 spec-level -- a scale, a softmax reduce and two matmuls emitted as \
                 separate nodes (proxima-tensor/src/spec.rs's mistral forward), so the \
                 scores matrix is materialized in full. -fa is a FUSED node: one \
                 op::Reduce whose body streams k/v tiles without ever writing the \
                 scores buffer. It attaches as a new fused ScalarOp/Reduce pair plus a \
                 cpu.rs kernel, next to the NeonTilePlan gate that already fuses \
                 multiply-into-add."
            ),
        }
    }

    /// KV cache element types.
    pub fn kv_cache_dtypes(&self) -> Option<(CacheDType, CacheDType)> {
        match (self.kv_cache_key_dtype, self.kv_cache_value_dtype) {
            (None, None) => None,
            _ => todo!(
                "kv cache quantization (llama.cpp -ctk/-ctv): there is NO kv cache to \
                 quantize. attention recomputes every position on every call; nothing \
                 in this crate retains k/v between forwards. The quantizers themselves \
                 already exist and are measured (proxima-tensor's Q8_K/Q4_K packing and \
                 dot_q4k_q8k), so this knob is the LAST step, not the first: it becomes \
                 real the moment a cache type exists to choose a codec for, and -ctk \
                 q8_0 is then a call into packing that already ships."
            ),
        }
    }

    /// Accelerator offload plan.
    pub fn offload_plan(&self) -> Option<(u32, bool)> {
        match (self.gpu_layers, self.kv_offload) {
            (None, None) => None,
            _ => todo!(
                "layer offload (llama.cpp -ngl / --no-kv-offload): no path from a bound \
                 program to a GPU executor. omega::metal::execute takes \
                 `blocks: &[&[f32]]`, i.e. dequantized f32 slices, so it cannot accept \
                 the packed QuantizedBlock weights this crate binds -- offloading a \
                 layer today would mean dequantizing it first, which spends the memory \
                 the flag exists to save. This attaches at the same seam as \
                 proxima_tensor::policy::Device::Metal: teach the Metal entry point \
                 packed operands, then -ngl selects how many layers take it."
            ),
        }
    }

    /// Generation-loop budget.
    pub fn reasoning_budget(&self) -> Option<u32> {
        match self.reasoning_budget_tokens {
            None => None,
            Some(_) => todo!(
                "reasoning budget (llama.cpp --reasoning-budget): there is no generation \
                 loop to budget. bind's harness runs one forward and greedy-picks one \
                 token; nothing samples repeatedly, so no counter can bound it. This \
                 attaches to whatever owns the sampling loop, alongside -c and the KV \
                 cache, and is the only knob in this list that needs no tensor work at \
                 all -- it is pure control flow above the forward."
            ),
        }
    }
}

#[cfg(feature = "config")]
impl ModelPolicy {
    /// Load an explicit path: compiled base, then the file, then set
    /// environment keys, then validation.
    pub fn load(path: impl AsRef<std::path::Path>) -> conflaguration::Result<Self> {
        ConfigBuilder::<Self>::new()
            .value(Self::COMPILED)
            .file(path)
            .env()
            .validate()
            .build()
    }

    /// Discovery form: `$PROXIMA_MODEL_CONFIG`, else `./proxima-model.toml`
    /// when it exists, else compiled-plus-environment.
    #[must_use]
    pub fn discover() -> Self {
        let chain = ConfigBuilder::<Self>::new().value(Self::COMPILED);
        let chain = match discovered_path() {
            Some(path) => chain.file(path),
            None => chain,
        };
        match chain.env().validate().build() {
            Ok(policy) => policy,
            Err(error) => {
                std::eprintln!(
                    "proxima-model-interop: model policy unusable ({error}); using compiled defaults"
                );
                Self::COMPILED
            }
        }
    }
}

#[cfg(feature = "config")]
fn discovered_path() -> Option<std::path::PathBuf> {
    if let Ok(explicit) = std::env::var("PROXIMA_MODEL_CONFIG") {
        return Some(std::path::PathBuf::from(explicit));
    }
    let fallback = std::path::PathBuf::from("proxima-model.toml");
    fallback.is_file().then_some(fallback)
}

#[cfg(feature = "config")]
impl Validate for ModelPolicy {
    fn validate(&self) -> conflaguration::Result<()> {
        Ok(())
    }
}

static ACTIVE: OnceLock<ModelPolicy> = OnceLock::new();

/// The process-wide model policy, resolved at most once.
#[must_use]
pub fn active() -> &'static ModelPolicy {
    ACTIVE.get_or_init(resolve)
}

/// Install programmatically before the first load or forward; fails, handing
/// the value back, if resolution already happened.
pub fn install(policy: ModelPolicy) -> Result<(), ModelPolicy> {
    ACTIVE.set(policy)
}

#[cfg(feature = "config")]
fn resolve() -> ModelPolicy {
    ModelPolicy::discover()
}

#[cfg(not(feature = "config"))]
fn resolve() -> ModelPolicy {
    ModelPolicy::COMPILED
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_todays_behaviour_and_reach_no_todo() {
        let policy = ModelPolicy::COMPILED;

        assert!(!policy.prefault);
        assert_eq!(policy.context_window(), None);
        assert_eq!(policy.batch_shape(), None);
        assert_eq!(policy.flash_attention_enabled(), None);
        assert_eq!(policy.kv_cache_dtypes(), None);
        assert_eq!(policy.offload_plan(), None);
        assert_eq!(policy.reasoning_budget(), None);
    }

    #[cfg(feature = "config")]
    #[test]
    fn a_toml_can_set_prefault_and_leaves_the_unimplemented_knobs_unset() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("proxima-model.toml");
        std::fs::write(&path, "prefault = true\n").expect("write toml");

        let policy = ModelPolicy::load(&path).expect("load the model policy");

        assert!(policy.prefault);
        assert_eq!(policy.kv_cache_key_dtype, None);
    }

    #[cfg(feature = "config")]
    #[test]
    fn builder_and_empty_config_agree_with_compiled() {
        let built = ModelPolicy::builder().build();
        let deserialized: ModelPolicy = toml::from_str("").expect("empty toml is all defaults");

        assert_eq!(built, ModelPolicy::COMPILED);
        assert_eq!(deserialized, ModelPolicy::COMPILED);
    }

    #[cfg(feature = "config")]
    #[test]
    #[should_panic(expected = "kv cache quantization")]
    fn asking_for_a_quantized_kv_cache_says_what_is_missing_instead_of_lying() {
        let policy = ModelPolicy {
            kv_cache_key_dtype: Some(CacheDType::Q8_0),
            ..ModelPolicy::COMPILED
        };

        let _ = policy.kv_cache_dtypes();
    }
}
