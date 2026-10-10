use bon::Builder;
use conflaguration::Settings;
#[cfg(all(
    feature = "metal",
    feature = "metal-attn-variants",
    target_os = "macos"
))]
use omega::{AttentionSimdgroupCount, AttentionTileHeight, AttentionVariant};
#[cfg(all(feature = "metal", target_os = "macos"))]
use omega::{DispatchType, MathMode};
use proxima_gguf::types::GgmlType;
use proxima_tensor::NumericPolicy;
use serde::{Deserialize, Serialize};

use crate::RopeScaling;
use crate::prompt_cache_settings::PromptCacheSettings;
use crate::serving::{
    ContextLength, DEFAULT_BATCH_SIZE, DEFAULT_GPU_LAYERS, DEFAULT_MODEL_PATH,
    DEFAULT_RESIDENT_PREFILL_PLAN_BYTES, DEFAULT_UBATCH_SIZE, GdnPrefillBackend, NamePattern,
    ServingConfig, WeightPrecisionRule,
};
use crate::speculative_settings::SpeculativeSettings;

mod levels;

pub use levels::{
    AdmissionScheduleSettings, ExpertResidencyScheduleSettings, PhaseScheduleSettings,
};

fn from_json<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> {
    serde_json::from_str(raw)
}

fn from_name<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> {
    serde_json::from_value(serde_json::Value::String(raw.to_owned()))
}

/// the values llama.cpp accepts for `--cache-type-k` and `--cache-type-v`, spelled
/// as llama.cpp spells them; `as_ggml` is the lowering to the type the serving
/// config holds. mirrored rather than derived on `GgmlType` because that enum
/// is foreign and `#[non_exhaustive]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheType {
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "bf16")]
    Bf16,
    #[serde(rename = "q8_0")]
    Q80,
    #[serde(rename = "q4_0")]
    Q40,
    #[serde(rename = "q4_1")]
    Q41,
    #[serde(rename = "q5_0")]
    Q50,
    #[serde(rename = "q5_1")]
    Q51,
    #[serde(rename = "iq4_nl")]
    Iq4Nl,
}

impl CacheType {
    pub const fn as_ggml(self) -> GgmlType {
        match self {
            Self::F32 => GgmlType::F32,
            Self::F16 => GgmlType::F16,
            Self::Bf16 => GgmlType::Bf16,
            Self::Q80 => GgmlType::Q8_0,
            Self::Q40 => GgmlType::Q4_0,
            Self::Q41 => GgmlType::Q4_1,
            Self::Q50 => GgmlType::Q5_0,
            Self::Q51 => GgmlType::Q5_1,
            Self::Iq4Nl => GgmlType::Iq4Nl,
        }
    }
}

/// the Metal math modes a serving configuration can request, spelled in
/// lowercase; `as_math_mode` is the lowering to the backend's own enum, which
/// is foreign and carries no serde derive.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MathModeName {
    Safe,
    Relaxed,
    Fast,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl MathModeName {
    pub const fn as_math_mode(self) -> MathMode {
        match self {
            Self::Safe => MathMode::Safe,
            Self::Relaxed => MathMode::Relaxed,
            Self::Fast => MathMode::Fast,
        }
    }
}

/// the Metal dispatch encodings a serving configuration can request, spelled in
/// lowercase; `as_dispatch_type` is the lowering to the backend's own enum,
/// which is foreign and carries no serde derive.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchTypeName {
    Serial,
    Concurrent,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl DispatchTypeName {
    pub const fn as_dispatch_type(self) -> DispatchType {
        match self {
            Self::Serial => DispatchType::Serial,
            Self::Concurrent => DispatchType::Concurrent,
        }
    }
}

/// one per-tensor recode rule as written in a config file: which tensor names
/// it matches (`pattern_kind` selects exact, prefix or suffix) and the type
/// those tensors are bound at. a target with no encoder is refused at bind time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "pattern_kind", rename_all = "snake_case")]
pub enum WeightPrecisionRuleSettings {
    Exact { pattern: String, target: CacheType },
    Prefix { pattern: String, target: CacheType },
    Suffix { pattern: String, target: CacheType },
}

/// the llama-flag surface of [`ServingConfig`], loadable from toml and env
/// (`PROXIMA_SERVING_<FIELD>`) and fluent-buildable; `as_serving_config` lowers
/// it into the `Copy` shape the decode path reads.
#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_SERVING")]
#[builder(derive(Clone, Debug))]
#[serde(default)]
pub struct ServingSettings {
    /// `ServingConfig::model_path`: `-m`, the model file.
    #[setting(default = "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf")]
    #[builder(default = DEFAULT_MODEL_PATH.to_owned())]
    pub model_path: String,
    /// `ServingConfig::context_length`: `-c`; `0` serves the model's native length.
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub context_length: u32,
    /// `ServingConfig::context_length`: serve `context_length` even past the trained limit.
    #[setting(default = false)]
    #[builder(default = false)]
    pub context_extrapolate: bool,
    /// `ServingConfig::rope_scaling`: `--rope-scaling` and its companions.
    #[setting(resolve_with = "from_json", default)]
    pub rope_scaling: Option<RopeScaling>,
    /// `ServingConfig::parallel_sequences`: `-np`.
    #[setting(default = 1)]
    #[builder(default = 1)]
    pub parallel_sequences: u32,
    /// `ServingConfig::kv_cache_key_quant`: `--cache-type-k`.
    #[setting(resolve_with = "from_name", default_str = "f32")]
    #[builder(default = CacheType::F32)]
    pub kv_cache_key_quant: CacheType,
    /// `ServingConfig::kv_cache_value_quant`: `--cache-type-v`.
    #[setting(resolve_with = "from_name", default_str = "f32")]
    #[builder(default = CacheType::F32)]
    pub kv_cache_value_quant: CacheType,
    /// `ServingConfig::flash_attention`: `-fa`.
    #[setting(default = false)]
    #[builder(default = false)]
    pub flash_attention: bool,
    /// `ServingConfig::batch_size`: `-b`.
    #[setting(default = 2048)]
    #[builder(default = DEFAULT_BATCH_SIZE)]
    pub batch_size: u32,
    /// `ServingConfig::ubatch_size`: `-ub`.
    #[setting(default = 512)]
    #[builder(default = DEFAULT_UBATCH_SIZE)]
    pub ubatch_size: u32,
    /// `ServingConfig::gpu_layers`: `-ngl`; `-1` offloads every layer.
    #[cfg_attr(feature = "metal", setting(default = -1))]
    #[cfg_attr(not(feature = "metal"), setting(default = 0))]
    #[builder(default = DEFAULT_GPU_LAYERS)]
    pub gpu_layers: i32,
    /// `ServingConfig::gpu_memory_fit`: `--fit`.
    #[setting(default = true)]
    #[builder(default = true)]
    pub gpu_memory_fit: bool,
    /// `ServingConfig::gpu_memory_limit_bytes`: `--fit-target`, in bytes.
    pub gpu_memory_limit_bytes: Option<u64>,
    /// `ServingConfig::kv_offload`: `--kv-offload`.
    #[setting(default = false)]
    #[builder(default = false)]
    pub kv_offload: bool,
    /// `ServingConfig::multimodal_projector`: `--mmproj`.
    #[setting(default = false)]
    #[builder(default = false)]
    pub multimodal_projector: bool,
    /// `ServingConfig::reasoning_budget`: `--reasoning-budget`.
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub reasoning_budget: i32,
    /// `ServingConfig::temperature`: `--temp`; `<= 0.0` samples greedily.
    #[setting(default = 0.0)]
    #[builder(default = 0.0)]
    pub temperature: f32,
    /// `ServingConfig::top_k`: `--top-k`; `<= 0` disables the filter.
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub top_k: i32,
    /// `ServingConfig::top_p`: `--top-p`; `1.0` disables the filter.
    #[setting(default = 1.0)]
    #[builder(default = 1.0)]
    pub top_p: f32,
    /// `ServingConfig::min_p`: `--min-p`; `0.0` disables the filter.
    #[setting(default = 0.0)]
    #[builder(default = 0.0)]
    pub min_p: f32,
    /// `ServingConfig::repeat_last_n`: `--repeat-last-n`, the penalty window in tokens.
    #[setting(default = 64)]
    #[builder(default = 64)]
    pub repeat_last_n: i32,
    /// `ServingConfig::repeat_penalty`: `--repeat-penalty`; `1.0` disables it.
    #[setting(default = 1.0)]
    #[builder(default = 1.0)]
    pub repeat_penalty: f32,
    /// `ServingConfig::frequency_penalty`: `--frequency-penalty`.
    #[setting(default = 0.0)]
    #[builder(default = 0.0)]
    pub frequency_penalty: f32,
    /// `ServingConfig::presence_penalty`: `--presence-penalty`.
    #[setting(default = 0.0)]
    #[builder(default = 0.0)]
    pub presence_penalty: f32,
    /// `ServingConfig::seed`: `--seed`, the sampler's random seed.
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub seed: u64,
    /// `ServingConfig::kv_bucket_tokens`: the key/value extent rounding step; `1` disables bucketing.
    #[setting(default = 32)]
    #[builder(default = 32)]
    pub kv_bucket_tokens: usize,
    /// `ServingConfig::math_mode`: the Metal math mode every compiled kernel uses.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[setting(resolve_with = "from_name", default_str = "relaxed")]
    #[builder(default = MathModeName::Relaxed)]
    pub math_mode: MathModeName,
    /// `ServingConfig::numeric_policy`: the rewrites the plan may apply, one permission per field.
    #[setting(
        resolve_with = "from_json",
        default_str = "{\"contraction\":true,\"reassociation\":true,\"nan_assumptions\":false,\"signed_zero\":false,\"approx_functions\":false,\"epilogue_sources\":true}"
    )]
    #[builder(default = NumericPolicy::llama_relaxed().with_epilogue_sources(true))]
    pub numeric_policy: NumericPolicy,
    /// `ServingConfig::exact_activations`: keep activations at full precision.
    #[setting(default = true)]
    #[builder(default = true)]
    pub exact_activations: bool,
    /// `ServingConfig::dispatch_type`: whether Metal dispatches run one at a time or may overlap.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[setting(resolve_with = "from_name", default_str = "serial")]
    #[builder(default = DispatchTypeName::Serial)]
    pub dispatch_type: DispatchTypeName,
    /// the simdgroup count for cached attention; `legacy` keeps the sized count.
    #[cfg(all(
        feature = "metal",
        feature = "metal-attn-variants",
        target_os = "macos"
    ))]
    #[setting(resolve_with = "from_name", default_str = "legacy")]
    #[builder(default = AttentionSimdgroupCount::Legacy)]
    pub attention_simdgroup_count: AttentionSimdgroupCount,
    /// row count assigned to each cached-attention threadgroup.
    #[cfg(all(
        feature = "metal",
        feature = "metal-attn-variants",
        target_os = "macos"
    ))]
    #[setting(resolve_with = "from_name", default_str = "legacy")]
    #[builder(default = AttentionTileHeightSetting::Legacy)]
    pub attention_tile_height: AttentionTileHeightSetting,
    /// `ServingConfig::weight_precision`: ordered per-tensor recode rules, first match wins.
    #[setting(resolve_with = "from_json", default_str = "[]")]
    #[builder(default)]
    pub weight_precision: Vec<WeightPrecisionRuleSettings>,
    /// `ServingConfig::gdn_prefill_backend`: where the gated delta net prefill runs.
    #[setting(resolve_with = "from_name", default_str = "cpu")]
    #[builder(default = GdnPrefillBackend::Cpu)]
    pub gdn_prefill_backend: GdnPrefillBackend,
    /// `ServingConfig::gpu_correctness_fallback`: run the CPU oracle even when the GPU is requested.
    #[setting(default = false)]
    #[builder(default = false)]
    pub gpu_correctness_fallback: bool,
    /// `ServingConfig::prefill_one_evaluation`: evaluate the whole prefill chunk in one call.
    #[setting(default = false)]
    #[builder(default = false)]
    pub prefill_one_evaluation: bool,
    /// `ServingConfig::prefill_chunk_positions`: positions per prefill chunk when one-call prefill is on; `0` keeps the unsplit behaviour.
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub prefill_chunk_positions: usize,
    /// `ServingConfig::cached_attention_fusion`: fuse the cached attention step.
    #[setting(default = true)]
    #[builder(default = true)]
    pub cached_attention_fusion: bool,
    /// `ServingConfig::gated_delta_net_fusion`: fuse the gated delta net step.
    #[setting(default = true)]
    #[builder(default = true)]
    pub gated_delta_net_fusion: bool,
    /// `ServingConfig::moe_topk_fusion`: fuse the expert top-k selection.
    #[setting(default = true)]
    #[builder(default = true)]
    pub moe_topk_fusion: bool,
    /// `ServingConfig::plan_time_constants`: keep plan-time constants resident.
    #[setting(default = true)]
    #[builder(default = true)]
    pub plan_time_constants: bool,
    /// `ServingConfig::plan_refit`: refit a compiled plan to a new extent instead of recompiling.
    #[setting(default = true)]
    #[builder(default = true)]
    pub plan_refit: bool,
    /// `ServingConfig::command_buffer_chunks`: command buffers one token's dispatches are split into; at least `1`.
    #[setting(default = 1)]
    #[builder(default = 1)]
    pub command_buffer_chunks: u32,
    /// `ServingConfig::max_command_buffers_per_token`: cap on command buffers per token; `0` is no cap.
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub max_command_buffers_per_token: usize,
    /// `ServingConfig::resident_prefill_plan_bytes`: bytes of output slots prompt-width plans may keep resident between requests; `0` frees them as soon as the shape changes.
    #[setting(default = 536870912)]
    #[builder(default = DEFAULT_RESIDENT_PREFILL_PLAN_BYTES)]
    pub resident_prefill_plan_bytes: usize,
    /// `ServingConfig::overlap_transfer_compute`: overlap weight transfer with compute.
    #[setting(default = false)]
    #[builder(default = false)]
    pub overlap_transfer_compute: bool,
    /// `ServingConfig::warm_model_buffers_at_load`: declare the loaded model's buffers to the driver at load instead of in the first prefill.
    #[setting(default = true)]
    #[builder(default = true)]
    pub warm_model_buffers_at_load: bool,
    #[setting(nested)]
    #[builder(default)]
    pub admission_schedule: AdmissionScheduleSettings,
    #[setting(nested)]
    #[builder(default)]
    pub phase_schedule: PhaseScheduleSettings,
    #[setting(nested)]
    #[builder(default)]
    pub expert_residency_schedule: ExpertResidencyScheduleSettings,
    #[setting(nested, override_prefix = "PROXIMA_SPECULATIVE")]
    #[builder(default = SpeculativeSettings::builder().build())]
    pub speculative: SpeculativeSettings,
    #[setting(nested, override_prefix = "PROXIMA_PROMPT_CACHE")]
    #[builder(default = PromptCacheSettings::builder().build())]
    pub prompt_cache: PromptCacheSettings,
}

impl Default for ServingSettings {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl ServingSettings {
    #[must_use]
    pub fn weight_precision_rules(&self) -> Vec<WeightPrecisionRule<'_>> {
        self.weight_precision
            .iter()
            .map(|rule| match rule {
                WeightPrecisionRuleSettings::Exact { pattern, target } => WeightPrecisionRule {
                    pattern: NamePattern::Exact(pattern),
                    target: target.as_ggml(),
                },
                WeightPrecisionRuleSettings::Prefix { pattern, target } => WeightPrecisionRule {
                    pattern: NamePattern::Prefix(pattern),
                    target: target.as_ggml(),
                },
                WeightPrecisionRuleSettings::Suffix { pattern, target } => WeightPrecisionRule {
                    pattern: NamePattern::Suffix(pattern),
                    target: target.as_ggml(),
                },
            })
            .collect()
    }

    #[must_use]
    pub fn as_serving_config<'a>(
        &'a self,
        weight_precision: &'a [WeightPrecisionRule<'a>],
    ) -> ServingConfig<'a> {
        ServingConfig {
            model_path: &self.model_path,
            context_length: self.lowered_context_length(),
            rope_scaling: self.rope_scaling,
            parallel_sequences: self.parallel_sequences,
            kv_cache_key_quant: self.kv_cache_key_quant.as_ggml(),
            kv_cache_value_quant: self.kv_cache_value_quant.as_ggml(),
            flash_attention: self.flash_attention,
            batch_size: self.batch_size,
            ubatch_size: self.ubatch_size,
            gpu_layers: self.gpu_layers,
            gpu_memory_fit: self.gpu_memory_fit,
            gpu_memory_limit_bytes: self.gpu_memory_limit_bytes,
            kv_offload: self.kv_offload,
            multimodal_projector: self.multimodal_projector,
            reasoning_budget: self.reasoning_budget,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repeat_last_n: self.repeat_last_n,
            repeat_penalty: self.repeat_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
            kv_bucket_tokens: self.kv_bucket_tokens,
            numeric_policy: self.numeric_policy,
            exact_activations: self.exact_activations,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            math_mode: self.math_mode.as_math_mode(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            dispatch_type: self.dispatch_type.as_dispatch_type(),
            #[cfg(all(
                feature = "metal",
                feature = "metal-attn-variants",
                target_os = "macos"
            ))]
            attention_variant: {
                let mut variant = AttentionVariant::default();
                variant.simdgroup_count = self.attention_simdgroup_count;
                variant.tile_height = self.attention_tile_height.as_tile_height();
                if variant.simdgroup_count == AttentionSimdgroupCount::Legacy
                    && variant.tile_height == AttentionTileHeight::Legacy
                {
                    None
                } else {
                    Some(variant)
                }
            },
            weight_precision,
            gdn_prefill_backend: self.gdn_prefill_backend,
            gpu_correctness_fallback: self.gpu_correctness_fallback,
            prefill_one_evaluation: self.prefill_one_evaluation,
            prefill_chunk_positions: self.prefill_chunk_positions,
            cached_attention_fusion: self.cached_attention_fusion,
            gated_delta_net_fusion: self.gated_delta_net_fusion,
            moe_topk_fusion: self.moe_topk_fusion,
            plan_time_constants: self.plan_time_constants,
            plan_refit: self.plan_refit,
            command_buffer_chunks: self.command_buffer_chunks,
            max_command_buffers_per_token: self.max_command_buffers_per_token,
            resident_prefill_plan_bytes: self.resident_prefill_plan_bytes,
            overlap_transfer_compute: self.overlap_transfer_compute,
            warm_model_buffers_at_load: self.warm_model_buffers_at_load,
            admission_schedule: self.admission_schedule.as_admission_schedule(),
            phase_schedule: self.phase_schedule.as_phase_schedule(),
            expert_residency_schedule: self.expert_residency_schedule.as_expert_residency_schedule(),
            speculative: self.speculative.as_speculative_config(),
            prompt_cache: self.prompt_cache.as_prompt_cache_config(),
            ..ServingConfig::default()
        }
    }

    const fn lowered_context_length(&self) -> ContextLength {
        match (self.context_length, self.context_extrapolate) {
            (0, _) => ContextLength::Native,
            (length, false) => ContextLength::Within(length),
            (length, true) => ContextLength::Extrapolate(length),
        }
    }
}

#[cfg(test)]
mod round_trip {
    use conflaguration::Settings;

    use super::ServingSettings;

    const SECTION_PREFIXES: [&str; 3] = [
        "PROXIMA_SERVING_",
        "PROXIMA_SPECULATIVE_",
        "PROXIMA_PROMPT_CACHE_",
    ];

    fn cleared_section_env() -> Vec<(String, Option<String>)> {
        std::env::vars_os()
            .filter_map(|(key, _)| key.into_string().ok())
            .filter(|key| SECTION_PREFIXES.iter().any(|prefix| key.starts_with(prefix)))
            .map(|key| (key, None))
            .collect()
    }

    pub(super) fn assert_three_ways(
        toml_text: &str,
        env_pairs: &[(&str, &str)],
        via_builder: &ServingSettings,
    ) {
        let parsed_toml: ServingSettings = conflaguration::from_toml_str(toml_text)
            .unwrap_or_else(|err| panic!("the toml loader failed: {err}"));
        let mut env = cleared_section_env();
        env.extend(
            env_pairs
                .iter()
                .map(|(key, value)| ((*key).to_owned(), Some((*value).to_owned()))),
        );
        let parsed_env = temp_env::with_vars(env, || {
            ServingSettings::from_env()
                .unwrap_or_else(|err| panic!("the env loader failed: {err}"))
        });

        assert_eq!(&parsed_toml, via_builder, "the toml loader differs from the builder");
        assert_eq!(&parsed_env, via_builder, "the env loader differs from the builder");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::speculative_settings::{SpeculativeTypeName, SpeculativeTypeNameSet};

    #[test]
    fn unset_batch_sizes_lower_to_the_serving_config_defaults_from_every_surface() {
        let from_builder = ServingSettings::builder().build();
        let from_empty_toml: ServingSettings =
            conflaguration::from_toml_str("").expect("an empty toml takes every default");

        for settings in [&from_builder, &from_empty_toml] {
            let lowered = settings.as_serving_config(&[]);
            assert_eq!(lowered.batch_size, ServingConfig::default().batch_size);
            assert_eq!(lowered.ubatch_size, ServingConfig::default().ubatch_size);
            assert_eq!(lowered.ubatch_size, 512, "llama.cpp's own -ub default");
        }
    }

    #[test]
    fn serving_section_cache_type_round_trips_json() {
        let table = [
            ("f32", CacheType::F32, GgmlType::F32),
            ("f16", CacheType::F16, GgmlType::F16),
            ("bf16", CacheType::Bf16, GgmlType::Bf16),
            ("q8_0", CacheType::Q80, GgmlType::Q8_0),
            ("q4_0", CacheType::Q40, GgmlType::Q4_0),
            ("q4_1", CacheType::Q41, GgmlType::Q4_1),
            ("q5_0", CacheType::Q50, GgmlType::Q5_0),
            ("q5_1", CacheType::Q51, GgmlType::Q5_1),
            ("iq4_nl", CacheType::Iq4Nl, GgmlType::Iq4Nl),
        ];
        assert_eq!(table.len(), 9, "the table must list every cache type");

        for (word, variant, ggml) in table {
            let quoted = format!("\"{word}\"");
            let parsed: CacheType = serde_json::from_str(&quoted)
                .unwrap_or_else(|error| panic!("{quoted} must parse: {error}"));
            assert_eq!(parsed, variant, "{quoted} parses to its variant");
            let written = serde_json::to_string(&variant)
                .unwrap_or_else(|error| panic!("{word} must serialize: {error}"));
            assert_eq!(written, quoted, "{word} serializes back to the same word");
            assert_eq!(variant.as_ggml(), ggml, "{word} lowers to its ggml type");
        }

        for rejected in ["\"q9_9\"", "\"Q8_0\"", "\"q8\""] {
            assert!(
                serde_json::from_str::<CacheType>(rejected).is_err(),
                "{rejected} must be refused"
            );
        }
    }

    const FIRST_FIFTEEN_TOML: &str = r#"
model_path = "/models/e2b-it-qat.gguf"
context_length = 32768
context_extrapolate = true
parallel_sequences = 2
kv_cache_key_quant = "q8_0"
kv_cache_value_quant = "q4_0"
flash_attention = true
batch_size = 512
ubatch_size = 128
gpu_layers = 99
gpu_memory_fit = false
gpu_memory_limit_bytes = 17179869184
kv_offload = true
multimodal_projector = true
reasoning_budget = 1024

[rope_scaling]
kind = "yarn"
factor = 4.0
original_context = 32768
extrapolation_factor = 1.0
attention_factor = 1.1386294
beta_fast = 32.0
beta_slow = 1.0
"#;

    const YARN_JSON: &str = r#"{"kind":"yarn","factor":4.0,"original_context":32768,"extrapolation_factor":1.0,"attention_factor":1.1386294,"beta_fast":32.0,"beta_slow":1.0}"#;

    fn yarn() -> RopeScaling {
        RopeScaling::Yarn {
            factor: 4.0,
            original_context: 32768,
            extrapolation_factor: 1.0,
            attention_factor: 1.1386294,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }

    #[test]
    fn serving_scalars_first_fifteen_lower_and_round_trip() {
        let from_toml: ServingSettings = conflaguration::from_toml_str(FIRST_FIFTEEN_TOML)
            .expect("the first-fifteen toml parses");
        let built = ServingSettings::builder()
            .model_path("/models/e2b-it-qat.gguf".to_owned())
            .context_length(32768)
            .context_extrapolate(true)
            .rope_scaling(yarn())
            .parallel_sequences(2)
            .kv_cache_key_quant(CacheType::Q80)
            .kv_cache_value_quant(CacheType::Q40)
            .flash_attention(true)
            .batch_size(512)
            .ubatch_size(128)
            .gpu_layers(99)
            .gpu_memory_fit(false)
            .gpu_memory_limit_bytes(17_179_869_184)
            .kv_offload(true)
            .multimodal_projector(true)
            .reasoning_budget(1024)
            .build();
        let from_env = temp_env::with_vars(
            [
                ("PROXIMA_SERVING_MODEL_PATH", Some("/models/e2b-it-qat.gguf")),
                ("PROXIMA_SERVING_CONTEXT_LENGTH", Some("32768")),
                ("PROXIMA_SERVING_CONTEXT_EXTRAPOLATE", Some("true")),
                ("PROXIMA_SERVING_PARALLEL_SEQUENCES", Some("2")),
                ("PROXIMA_SERVING_KV_CACHE_KEY_QUANT", Some("q8_0")),
                ("PROXIMA_SERVING_KV_CACHE_VALUE_QUANT", Some("q4_0")),
                ("PROXIMA_SERVING_FLASH_ATTENTION", Some("true")),
                ("PROXIMA_SERVING_BATCH_SIZE", Some("512")),
                ("PROXIMA_SERVING_UBATCH_SIZE", Some("128")),
                ("PROXIMA_SERVING_GPU_LAYERS", Some("99")),
                ("PROXIMA_SERVING_GPU_MEMORY_FIT", Some("false")),
                ("PROXIMA_SERVING_GPU_MEMORY_LIMIT_BYTES", Some("17179869184")),
                ("PROXIMA_SERVING_KV_OFFLOAD", Some("true")),
                ("PROXIMA_SERVING_MULTIMODAL_PROJECTOR", Some("true")),
                ("PROXIMA_SERVING_REASONING_BUDGET", Some("1024")),
                ("PROXIMA_SERVING_ROPE_SCALING", Some(YARN_JSON)),
            ],
            || ServingSettings::from_env().expect("the first-fifteen env parses"),
        );

        assert_eq!(from_toml, built, "toml and builder agree");
        assert_eq!(from_env, built, "env and builder agree");

        let lowered = built.as_serving_config(&[]);
        assert_eq!(lowered.context_length, ContextLength::Extrapolate(32768));
        assert_eq!(lowered.kv_cache_key_quant, GgmlType::Q8_0);
        assert_eq!(lowered.kv_cache_value_quant, GgmlType::Q4_0);
        assert_eq!(lowered.gpu_layers, 99);
        assert_eq!(lowered.gpu_memory_limit_bytes, Some(17_179_869_184));
        assert_eq!(lowered.rope_scaling, Some(yarn()));
        assert_eq!(lowered.model_path, "/models/e2b-it-qat.gguf");

        assert!(
            conflaguration::from_toml_str::<ServingSettings>(r#"kv_cache_key_quant = "q9_9""#)
                .is_err(),
            "an unknown cache type is refused"
        );
    }

    const SAMPLING_TOML: &str = r#"
temperature = 0.7
top_k = 40
top_p = 0.95
min_p = 0.05
repeat_last_n = 128
repeat_penalty = 1.1
frequency_penalty = 0.2
presence_penalty = 0.3
seed = 424242
kv_bucket_tokens = 64
exact_activations = false
MATH_MODE_LINE
[numeric_policy]
contraction = true
reassociation = false
nan_assumptions = true
signed_zero = true
approx_functions = true
epilogue_sources = true
"#;

    const POLICY_JSON: &str = r#"{"contraction":true,"reassociation":false,"nan_assumptions":true,"signed_zero":true,"approx_functions":true,"epilogue_sources":true}"#;

    fn granted_policy() -> NumericPolicy {
        let mut policy = NumericPolicy::bit_exact();
        policy.contraction = true;
        policy.reassociation = false;
        policy.nan_assumptions = true;
        policy.signed_zero = true;
        policy.approx_functions = true;
        policy.epilogue_sources = true;
        policy
    }

    const SAMPLING_ENV_KEYS: [&str; 13] = [
        "PROXIMA_SERVING_TEMPERATURE",
        "PROXIMA_SERVING_TOP_K",
        "PROXIMA_SERVING_TOP_P",
        "PROXIMA_SERVING_MIN_P",
        "PROXIMA_SERVING_REPEAT_LAST_N",
        "PROXIMA_SERVING_REPEAT_PENALTY",
        "PROXIMA_SERVING_FREQUENCY_PENALTY",
        "PROXIMA_SERVING_PRESENCE_PENALTY",
        "PROXIMA_SERVING_SEED",
        "PROXIMA_SERVING_KV_BUCKET_TOKENS",
        "PROXIMA_SERVING_EXACT_ACTIVATIONS",
        "PROXIMA_SERVING_NUMERIC_POLICY",
        "PROXIMA_SERVING_MATH_MODE",
    ];

    #[test]
    fn serving_scalars_sampling_and_policy_lower_and_round_trip() {
        let math_mode_line = if cfg!(all(feature = "metal", target_os = "macos")) {
            "math_mode = \"fast\""
        } else {
            ""
        };
        let toml = SAMPLING_TOML.replace("MATH_MODE_LINE", math_mode_line);
        let from_toml: ServingSettings =
            conflaguration::from_toml_str(&toml).expect("the sampling toml parses");
        let builder = ServingSettings::builder()
            .temperature(0.7)
            .top_k(40)
            .top_p(0.95)
            .min_p(0.05)
            .repeat_last_n(128)
            .repeat_penalty(1.1)
            .frequency_penalty(0.2)
            .presence_penalty(0.3)
            .seed(424_242)
            .kv_bucket_tokens(64)
            .exact_activations(false)
            .numeric_policy(granted_policy());
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let builder = builder.math_mode(MathModeName::Fast);
        let built = builder.build();
        let from_env = temp_env::with_vars(
            [
                ("PROXIMA_SERVING_TEMPERATURE", Some("0.7")),
                ("PROXIMA_SERVING_TOP_K", Some("40")),
                ("PROXIMA_SERVING_TOP_P", Some("0.95")),
                ("PROXIMA_SERVING_MIN_P", Some("0.05")),
                ("PROXIMA_SERVING_REPEAT_LAST_N", Some("128")),
                ("PROXIMA_SERVING_REPEAT_PENALTY", Some("1.1")),
                ("PROXIMA_SERVING_FREQUENCY_PENALTY", Some("0.2")),
                ("PROXIMA_SERVING_PRESENCE_PENALTY", Some("0.3")),
                ("PROXIMA_SERVING_SEED", Some("424242")),
                ("PROXIMA_SERVING_KV_BUCKET_TOKENS", Some("64")),
                ("PROXIMA_SERVING_EXACT_ACTIVATIONS", Some("false")),
                ("PROXIMA_SERVING_NUMERIC_POLICY", Some(POLICY_JSON)),
                (
                    "PROXIMA_SERVING_MATH_MODE",
                    cfg!(all(feature = "metal", target_os = "macos")).then_some("fast"),
                ),
            ],
            || ServingSettings::from_env().expect("the sampling env parses"),
        );

        assert_eq!(from_toml, built, "toml and builder agree");
        assert_eq!(from_env, built, "env and builder agree");

        let lowered = built.as_serving_config(&[]);
        assert_eq!(lowered.temperature, 0.7);
        assert_eq!(lowered.top_k, 40);
        assert_eq!(lowered.seed, 424_242);
        assert_eq!(lowered.kv_bucket_tokens, 64);
        assert!(!lowered.exact_activations);
        assert!(lowered.numeric_policy.nan_assumptions);
        #[cfg(all(feature = "metal", target_os = "macos"))]
        assert_eq!(lowered.math_mode, MathMode::Fast);

        let defaults = temp_env::with_vars(
            SAMPLING_ENV_KEYS.map(|key| (key, None::<&str>)),
            || ServingSettings::from_env().expect("the unset env resolves to defaults"),
        );
        assert_eq!(
            defaults.numeric_policy,
            NumericPolicy::llama_relaxed().with_epilogue_sources(true)
        );
        assert_eq!(defaults.kv_bucket_tokens, 32);
        assert_eq!(defaults, ServingSettings::default());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn serving_scalars_dispatch_type_lowers_and_round_trips() {
        let from_toml: ServingSettings = conflaguration::from_toml_str("dispatch_type = \"concurrent\"")
            .expect("the dispatch type toml parses");
        let built = ServingSettings::builder()
            .dispatch_type(DispatchTypeName::Concurrent)
            .build();
        let from_env = temp_env::with_vars(
            [("PROXIMA_SERVING_DISPATCH_TYPE", Some("concurrent"))],
            || ServingSettings::from_env().expect("the dispatch type env parses"),
        );

        assert_eq!(from_toml, built, "toml and builder agree");
        assert_eq!(from_env, built, "env and builder agree");
        assert_eq!(
            built.as_serving_config(&[]).dispatch_type,
            DispatchType::Concurrent
        );
        assert_eq!(
            ServingSettings::default().as_serving_config(&[]).dispatch_type,
            DispatchType::Serial
        );
        let refused = conflaguration::from_toml_str::<ServingSettings>("dispatch_type = \"parallel\"");
        assert!(refused.is_err(), "an unknown dispatch type is refused");
    }

    #[cfg(all(
        feature = "metal",
        feature = "metal-attn-variants",
        target_os = "macos"
    ))]
    #[test]
    fn simdgroup_count_setting_round_trips_and_reaches_serving_config() {
        let selections = [
            ("groups2", AttentionSimdgroupCount::Groups2),
            ("groups4", AttentionSimdgroupCount::Groups4),
            ("groups8", AttentionSimdgroupCount::Groups8),
        ];

        for (name, simdgroup_count) in selections {
            let setting = format!("attention_simdgroup_count = \"{name}\"");
            let built = ServingSettings::builder()
                .attention_simdgroup_count(simdgroup_count)
                .build();
            round_trip::assert_three_ways(
                &setting,
                &[("PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT", name)],
                &built,
            );

            let lowered = built
                .as_serving_config(&[])
                .attention_variant
                .expect("an explicit simdgroup count lowers to an attention variant");
            assert_eq!(lowered.simdgroup_count, simdgroup_count);
            assert_eq!(lowered.mma_precision, omega::AttentionMmaPrecision::Legacy);
        }

        let default = ServingSettings::default();
        assert_eq!(default.attention_simdgroup_count, AttentionSimdgroupCount::Legacy);
        assert!(
            default.as_serving_config(&[]).attention_variant.is_none(),
            "legacy count leaves the sized attention selection in place"
        );

        let invalid: Result<ServingSettings, _> =
            conflaguration::from_toml_str("attention_simdgroup_count = \"groups16\"");
        assert!(invalid.is_err(), "unsupported simdgroup counts are rejected");
    }

    #[cfg(all(
        feature = "metal",
        feature = "metal-attn-variants",
        target_os = "macos"
    ))]
    #[test]
    fn attention_tile_height_setting_round_trips_and_reaches_serving_config() {
        for (name, tile_height) in [
            ("rows8", AttentionTileHeightSetting::Rows8),
            ("rows16", AttentionTileHeightSetting::Rows16),
        ] {
            let setting = format!(
                "attention_simdgroup_count = \"groups4\"\nattention_tile_height = \"{name}\""
            );
            let built = ServingSettings::builder()
                .attention_simdgroup_count(AttentionSimdgroupCount::Groups4)
                .attention_tile_height(tile_height)
                .build();
            round_trip::assert_three_ways(
                &setting,
                &[
                    ("PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT", "groups4"),
                    ("PROXIMA_SERVING_ATTENTION_TILE_HEIGHT", name),
                ],
                &built,
            );

            let lowered = built
                .as_serving_config(&[])
                .attention_variant
                .expect("an explicit tile height lowers to an attention variant");
            assert_eq!(lowered.tile_height, tile_height.as_tile_height());
            assert_eq!(lowered.simdgroup_count, AttentionSimdgroupCount::Groups4);
            assert_eq!(lowered.mma_precision, omega::AttentionMmaPrecision::Legacy);
        }

        let default = ServingSettings::default();
        assert_eq!(default.attention_tile_height, AttentionTileHeightSetting::Legacy);
        assert!(
            default.as_serving_config(&[]).attention_variant.is_none(),
            "legacy tile height and count preserve the default dispatch"
        );

        let invalid: Result<ServingSettings, _> =
            conflaguration::from_toml_str("attention_tile_height = \"rows32\"");
        assert!(invalid.is_err(), "unsupported tile heights are rejected");
    }


    #[test]
    fn serving_scalars_weight_precision_and_fallbacks_lower_and_round_trip() {
        const WEIGHT_PRECISION_TOML: &str = r#"
gdn_prefill_backend = "mlx"
gpu_correctness_fallback = true

[[weight_precision]]
pattern_kind = "suffix"
pattern = "ffn_down_exps.weight"
target = "q4_0"

[[weight_precision]]
pattern_kind = "exact"
pattern = "token_embd.weight"
target = "q8_0"
"#;
        const WEIGHT_PRECISION_JSON: &str = r#"[{"pattern_kind":"suffix","pattern":"ffn_down_exps.weight","target":"q4_0"},{"pattern_kind":"exact","pattern":"token_embd.weight","target":"q8_0"}]"#;

        let from_toml: ServingSettings = conflaguration::from_toml_str(WEIGHT_PRECISION_TOML)
            .expect("the weight precision toml parses");
        let built = ServingSettings::builder()
            .weight_precision(vec![
                WeightPrecisionRuleSettings::Suffix {
                    pattern: "ffn_down_exps.weight".to_owned(),
                    target: CacheType::Q40,
                },
                WeightPrecisionRuleSettings::Exact {
                    pattern: "token_embd.weight".to_owned(),
                    target: CacheType::Q80,
                },
            ])
            .gdn_prefill_backend(GdnPrefillBackend::Mlx)
            .gpu_correctness_fallback(true)
            .build();
        let from_env = temp_env::with_vars(
            [
                ("PROXIMA_SERVING_WEIGHT_PRECISION", Some(WEIGHT_PRECISION_JSON)),
                ("PROXIMA_SERVING_GDN_PREFILL_BACKEND", Some("mlx")),
                ("PROXIMA_SERVING_GPU_CORRECTNESS_FALLBACK", Some("true")),
            ],
            || ServingSettings::from_env().expect("the weight precision env parses"),
        );

        assert_eq!(from_toml, built, "toml and builder agree");
        assert_eq!(from_env, built, "env and builder agree");

        let rules = built.weight_precision_rules();
        let config = built.as_serving_config(&rules);
        assert_eq!(
            config.weight_precision,
            [
                WeightPrecisionRule {
                    pattern: NamePattern::Suffix("ffn_down_exps.weight"),
                    target: GgmlType::Q4_0,
                },
                WeightPrecisionRule {
                    pattern: NamePattern::Exact("token_embd.weight"),
                    target: GgmlType::Q8_0,
                },
            ]
        );
        assert_eq!(config.gdn_prefill_backend, GdnPrefillBackend::Mlx);
        assert!(config.gpu_correctness_fallback);

        let unknown_kind = conflaguration::from_toml_str::<ServingSettings>(
            "[[weight_precision]]\npattern_kind = \"glob\"\npattern = \"blk.*\"\ntarget = \"q4_0\"",
        );
        assert!(unknown_kind.is_err(), "an unknown pattern kind is refused");
        let missing_pattern = conflaguration::from_toml_str::<ServingSettings>(
            "[[weight_precision]]\npattern_kind = \"exact\"\ntarget = \"q4_0\"",
        );
        assert!(missing_pattern.is_err(), "an exact rule without a pattern is refused");
    }

    #[test]
    fn serving_scalars_runtime_switches_lower_and_round_trip() {
        const SWITCHES_TOML: &str = r#"
prefill_one_evaluation = true
prefill_chunk_positions = 256
cached_attention_fusion = false
gated_delta_net_fusion = false
moe_topk_fusion = false
plan_time_constants = false
plan_refit = false
command_buffer_chunks = 8
max_command_buffers_per_token = 12
resident_prefill_plan_bytes = 805306368
overlap_transfer_compute = true
warm_model_buffers_at_load = false
"#;

        let from_toml: ServingSettings =
            conflaguration::from_toml_str(SWITCHES_TOML).expect("the switches toml parses");
        let built = ServingSettings::builder()
            .prefill_one_evaluation(true)
            .prefill_chunk_positions(256)
            .cached_attention_fusion(false)
            .gated_delta_net_fusion(false)
            .moe_topk_fusion(false)
            .plan_time_constants(false)
            .plan_refit(false)
            .command_buffer_chunks(8)
            .max_command_buffers_per_token(12)
            .resident_prefill_plan_bytes(805_306_368)
            .overlap_transfer_compute(true)
            .warm_model_buffers_at_load(false)
            .build();
        let from_env = temp_env::with_vars(
            [
                ("PROXIMA_SERVING_PREFILL_ONE_EVALUATION", Some("true")),
                ("PROXIMA_SERVING_PREFILL_CHUNK_POSITIONS", Some("256")),
                ("PROXIMA_SERVING_CACHED_ATTENTION_FUSION", Some("false")),
                ("PROXIMA_SERVING_GATED_DELTA_NET_FUSION", Some("false")),
                ("PROXIMA_SERVING_MOE_TOPK_FUSION", Some("false")),
                ("PROXIMA_SERVING_PLAN_TIME_CONSTANTS", Some("false")),
                ("PROXIMA_SERVING_PLAN_REFIT", Some("false")),
                ("PROXIMA_SERVING_COMMAND_BUFFER_CHUNKS", Some("8")),
                ("PROXIMA_SERVING_MAX_COMMAND_BUFFERS_PER_TOKEN", Some("12")),
                ("PROXIMA_SERVING_RESIDENT_PREFILL_PLAN_BYTES", Some("805306368")),
                ("PROXIMA_SERVING_OVERLAP_TRANSFER_COMPUTE", Some("true")),
                ("PROXIMA_SERVING_WARM_MODEL_BUFFERS_AT_LOAD", Some("false")),
            ],
            || ServingSettings::from_env().expect("the switches env parses"),
        );

        assert_eq!(from_toml, built, "toml and builder agree");
        assert_eq!(from_env, built, "env and builder agree");

        let config = built.as_serving_config(&[]);
        assert!(config.prefill_one_evaluation);
        assert_eq!(config.prefill_chunk_positions, 256);
        assert!(!config.cached_attention_fusion);
        assert!(!config.gated_delta_net_fusion);
        assert!(!config.moe_topk_fusion);
        assert!(!config.plan_time_constants);
        assert!(!config.plan_refit);
        assert_eq!(config.command_buffer_chunks, 8);
        assert_eq!(config.max_command_buffers_per_token, 12);
        assert_eq!(config.resident_prefill_plan_bytes, 805_306_368);
        assert!(config.overlap_transfer_compute);
        assert!(!config.warm_model_buffers_at_load);

        let defaults = ServingSettings::default();
        let lowered = defaults.as_serving_config(&[]);
        let today = ServingConfig::default();
        assert_eq!(lowered.prefill_one_evaluation, today.prefill_one_evaluation);
        assert_eq!(lowered.prefill_chunk_positions, today.prefill_chunk_positions);
        assert_eq!(lowered.cached_attention_fusion, today.cached_attention_fusion);
        assert_eq!(lowered.gated_delta_net_fusion, today.gated_delta_net_fusion);
        assert_eq!(lowered.moe_topk_fusion, today.moe_topk_fusion);
        assert_eq!(lowered.plan_time_constants, today.plan_time_constants);
        assert_eq!(lowered.plan_refit, today.plan_refit);
        assert_eq!(lowered.command_buffer_chunks, today.command_buffer_chunks);
        assert_eq!(lowered.max_command_buffers_per_token, today.max_command_buffers_per_token);
        assert_eq!(lowered.resident_prefill_plan_bytes, today.resident_prefill_plan_bytes);
        assert_eq!(lowered.overlap_transfer_compute, today.overlap_transfer_compute);
        assert_eq!(lowered.warm_model_buffers_at_load, today.warm_model_buffers_at_load);
        assert!(today.warm_model_buffers_at_load, "the warm-up is on by default");
    }

    #[test]
    fn serving_scalars_admission_level_round_trips() {
        let from_toml: ServingSettings =
            conflaguration::from_toml_str("[admission_schedule]\nmax_concurrent_requests = 4\n")
                .expect("the admission toml parses");
        let built = ServingSettings::builder()
            .admission_schedule(
                AdmissionScheduleSettings::builder()
                    .max_concurrent_requests(4)
                    .build(),
            )
            .build();
        let from_env = temp_env::with_vars(
            [(
                "PROXIMA_SERVING_ADMISSION_SCHEDULE_MAX_CONCURRENT_REQUESTS",
                Some("4"),
            )],
            || ServingSettings::from_env().expect("the admission env parses"),
        );

        assert_eq!(from_toml, built, "toml and builder agree");
        assert_eq!(from_env, built, "env and builder agree");
        assert_eq!(built.as_serving_config(&[]).admission_schedule.max_concurrent_requests, 4);
        assert_eq!(
            ServingSettings::default().as_serving_config(&[]).admission_schedule,
            ServingConfig::default().admission_schedule,
        );

        let negative = conflaguration::from_toml_str::<ServingSettings>(
            "[admission_schedule]\nmax_concurrent_requests = -1\n",
        );
        assert!(negative.is_err(), "a negative ceiling is refused");
    }

    #[test]
    fn serving_scalars_phase_level_round_trips() {
        let from_toml: ServingSettings =
            conflaguration::from_toml_str("[phase_schedule]\nprefill_before_decode = false\n")
                .expect("the phase toml parses");
        let built = ServingSettings::builder()
            .phase_schedule(
                PhaseScheduleSettings::builder()
                    .prefill_before_decode(false)
                    .build(),
            )
            .build();
        let from_env = temp_env::with_vars(
            [(
                "PROXIMA_SERVING_PHASE_SCHEDULE_PREFILL_BEFORE_DECODE",
                Some("false"),
            )],
            || ServingSettings::from_env().expect("the phase env parses"),
        );

        assert_eq!(from_toml, built, "toml and builder agree");
        assert_eq!(from_env, built, "env and builder agree");
        assert!(!built.as_serving_config(&[]).phase_schedule.prefill_before_decode);
        assert_eq!(
            ServingSettings::default().as_serving_config(&[]).phase_schedule,
            ServingConfig::default().phase_schedule,
        );
    }

    #[test]
    fn serving_scalars_expert_residency_level_round_trips() {
        let from_toml: ServingSettings = conflaguration::from_toml_str(
            "[expert_residency_schedule]\nper_layer_budget_bytes = 268435456\n",
        )
        .expect("the expert residency toml parses");
        let built = ServingSettings::builder()
            .expert_residency_schedule(
                ExpertResidencyScheduleSettings::builder()
                    .per_layer_budget_bytes(268_435_456)
                    .build(),
            )
            .build();
        let from_env = temp_env::with_vars(
            [(
                "PROXIMA_SERVING_EXPERT_RESIDENCY_SCHEDULE_PER_LAYER_BUDGET_BYTES",
                Some("268435456"),
            )],
            || ServingSettings::from_env().expect("the expert residency env parses"),
        );

        assert_eq!(from_toml, built, "toml and builder agree");
        assert_eq!(from_env, built, "env and builder agree");
        assert_eq!(
            built
                .as_serving_config(&[])
                .expert_residency_schedule
                .per_layer_budget_bytes,
            268_435_456
        );
        assert_eq!(
            ServingSettings::default()
                .as_serving_config(&[])
                .expert_residency_schedule,
            ServingConfig::default().expert_residency_schedule,
        );
    }

    #[test]
    fn serving_scalars_speculative_and_prompt_cache_round_trip() {
        let toml_text = "[prompt_cache]\nbyte_budget = 1073741824\nmax_entries = 8\n\
ring_rewind_slack = 512\ncheckpoint_interval = 1024\nmax_checkpoints = 4\n\
cache_reuse_min = 64\nprewarm_chunk_tokens = 128\nfollow_up_branches = 3\n\
follow_up_max_tokens = 64\nfollow_up_temperature_milli = 700\n\
min_similarity_milli = 250\nblock_tokens = 32\nseal_horizon_rows = 256\nbloom_bits_per_entry = 8192\n\
bloom_hashes = 6\n\n[speculative]\nspeculative_types = \"ngram-simple,ngram-map-k\"\n\
n_max = 3\nn_min = 0\np_min = 0.0\nngram_simple_size_n = 16\nngram_simple_size_m = 32\n\
ngram_simple_min_hits = 2\nngram_map_k_size_n = 12\nngram_map_k_size_m = 48\n\
ngram_map_k_min_hits = 1\nngram_map_k4v_size_n = 12\nngram_map_k4v_size_m = 48\n\
ngram_map_k4v_min_hits = 1\nngram_mod_n_match = 24\nngram_mod_n_max = 64\n\
ngram_mod_n_min = 48\n";
        let env_pairs = [
            ("PROXIMA_PROMPT_CACHE_BYTE_BUDGET", "1073741824"),
            ("PROXIMA_PROMPT_CACHE_MAX_ENTRIES", "8"),
            ("PROXIMA_PROMPT_CACHE_RING_REWIND_SLACK", "512"),
            ("PROXIMA_PROMPT_CACHE_CHECKPOINT_INTERVAL", "1024"),
            ("PROXIMA_PROMPT_CACHE_MAX_CHECKPOINTS", "4"),
            ("PROXIMA_PROMPT_CACHE_CACHE_REUSE_MIN", "64"),
            ("PROXIMA_PROMPT_CACHE_PREWARM_CHUNK_TOKENS", "128"),
            ("PROXIMA_PROMPT_CACHE_FOLLOW_UP_BRANCHES", "3"),
            ("PROXIMA_PROMPT_CACHE_FOLLOW_UP_MAX_TOKENS", "64"),
            ("PROXIMA_PROMPT_CACHE_FOLLOW_UP_TEMPERATURE_MILLI", "700"),
            ("PROXIMA_PROMPT_CACHE_MIN_SIMILARITY_MILLI", "250"),
            ("PROXIMA_PROMPT_CACHE_BLOCK_TOKENS", "32"),
            ("PROXIMA_PROMPT_CACHE_BLOOM_BITS_PER_ENTRY", "8192"),
            ("PROXIMA_PROMPT_CACHE_BLOOM_HASHES", "6"),
            ("PROXIMA_SPECULATIVE_TYPES", "ngram-simple,ngram-map-k"),
            ("PROXIMA_SPECULATIVE_N_MAX", "3"),
            ("PROXIMA_SPECULATIVE_N_MIN", "0"),
            ("PROXIMA_SPECULATIVE_P_MIN", "0.0"),
            ("PROXIMA_SPECULATIVE_NGRAM_SIMPLE_SIZE_N", "16"),
            ("PROXIMA_SPECULATIVE_NGRAM_SIMPLE_SIZE_M", "32"),
            ("PROXIMA_SPECULATIVE_NGRAM_SIMPLE_MIN_HITS", "2"),
            ("PROXIMA_SPECULATIVE_NGRAM_MAP_K_SIZE_N", "12"),
            ("PROXIMA_SPECULATIVE_NGRAM_MAP_K_SIZE_M", "48"),
            ("PROXIMA_SPECULATIVE_NGRAM_MAP_K_MIN_HITS", "1"),
            ("PROXIMA_SPECULATIVE_NGRAM_MAP_K4V_SIZE_N", "12"),
            ("PROXIMA_SPECULATIVE_NGRAM_MAP_K4V_SIZE_M", "48"),
            ("PROXIMA_SPECULATIVE_NGRAM_MAP_K4V_MIN_HITS", "1"),
            ("PROXIMA_SPECULATIVE_NGRAM_MOD_N_MATCH", "24"),
            ("PROXIMA_SPECULATIVE_NGRAM_MOD_N_MAX", "64"),
            ("PROXIMA_SPECULATIVE_NGRAM_MOD_N_MIN", "48"),
        ];
        let via_builder = ServingSettings::builder()
            .prompt_cache(
                PromptCacheSettings::builder()
                    .byte_budget(1_073_741_824)
                    .max_entries(8)
                    .ring_rewind_slack(512)
                    .checkpoint_interval(1024)
                    .max_checkpoints(4)
                    .cache_reuse_min(64)
                    .prewarm_chunk_tokens(128)
                    .follow_up_branches(3)
                    .follow_up_max_tokens(64)
                    .follow_up_temperature_milli(700)
                    .min_similarity_milli(250)
                    .block_tokens(32)
                    .bloom_bits_per_entry(8192)
                    .bloom_hashes(6)
                    .build(),
            )
            .speculative(
                SpeculativeSettings::builder()
                    .speculative_types(
                        SpeculativeTypeNameSet::empty()
                            .insert(SpeculativeTypeName::NgramSimple)
                            .insert(SpeculativeTypeName::NgramMapK),
                    )
                    .ngram_simple_size_n(16)
                    .ngram_simple_size_m(32)
                    .ngram_simple_min_hits(2)
                    .build(),
            )
            .build();

        round_trip::assert_three_ways(toml_text, &env_pairs, &via_builder);

        let lowered = via_builder.as_serving_config(&[]);
        assert_eq!(lowered.prompt_cache.byte_budget, 1_073_741_824);
        assert_eq!(lowered.speculative.ngram_simple.size_n, 16);

        let off_switch = temp_env::with_vars(
            [("PROXIMA_PROMPT_CACHE_BYTE_BUDGET", Some("0"))],
            || ServingSettings::from_env().expect("the off switch env parses"),
        );
        assert!(!off_switch.as_serving_config(&[]).prompt_cache.is_enabled());

        let refused = conflaguration::from_toml_str::<ServingSettings>(
            "[prompt_cache]\nbyte_budget = \"large\"\n",
        );
        assert!(refused.is_err(), "a string byte budget is refused");
    }
}
