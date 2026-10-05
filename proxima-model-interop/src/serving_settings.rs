use bon::Builder;
use conflaguration::Settings;
use proxima_gguf::types::GgmlType;
use serde::{Deserialize, Serialize};

use crate::RopeScaling;
use crate::serving::{
    ContextLength, DEFAULT_GPU_LAYERS, DEFAULT_MODEL_PATH, ServingConfig, WeightPrecisionRule,
};

fn from_json<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> {
    serde_json::from_str(raw)
}

fn from_name<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> {
    serde_json::from_value(serde_json::Value::String(raw.to_owned()))
}

/// the values llama accepts for `--cache-type-k` and `--cache-type-v`, spelled
/// as llama spells them; `as_ggml` is the lowering to the type the serving
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
    #[setting(default = 32)]
    #[builder(default = 32)]
    pub batch_size: u32,
    /// `ServingConfig::ubatch_size`: `-ub`.
    #[setting(default = 32)]
    #[builder(default = 32)]
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
}

impl Default for ServingSettings {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl ServingSettings {
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
            weight_precision,
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

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
model_path = "/models/gemma4-e2b-it-qat.gguf"
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
            .model_path("/models/gemma4-e2b-it-qat.gguf".to_owned())
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
                ("PROXIMA_SERVING_MODEL_PATH", Some("/models/gemma4-e2b-it-qat.gguf")),
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
        assert_eq!(lowered.model_path, "/models/gemma4-e2b-it-qat.gguf");

        assert!(
            conflaguration::from_toml_str::<ServingSettings>(r#"kv_cache_key_quant = "q9_9""#)
                .is_err(),
            "an unknown cache type is refused"
        );
    }
}
