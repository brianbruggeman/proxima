//! HF `config.json` writer for the MLX export, the inverse of
//! [`crate::hf_config::parse_hf_config`] and [`crate::hf_config::architecture_from_hf_config`].
//!
//! Sans-IO: [`hf_config_bytes`] takes the [`ModelHparams`] that
//! [`crate::bind::architecture_from_metadata`] read from the GGUF and returns the
//! file's bytes; the edge writes them. [`copy_config`] is the HF-directory
//! counterpart, which carries the source file through untouched. Every key
//! written is drawn from [`HF_CONFIG_KEYS`], the list the reader consults, so a
//! field renamed on one side is a compile or test failure on the other.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde_json::{Map, Number, Value, json};

use crate::bind::ModelHparams;
use crate::error::InteropError;
use crate::hf_config::HF_CONFIG_KEYS;
use crate::sliding_pattern::hparams::SlidingPatternHparams;

/// How one GGUF `general.architecture` spells itself in an HF `config.json`.
/// `nested` names the sub-object (and its `model_type`) a multimodal
/// checkpoint keeps its text fields under; the fields are also written at the
/// top level because `parse_hf_config` reads only the top level.
struct FamilyLayout {
    gguf_architecture: &'static str,
    model_type: &'static str,
    architecture_class: &'static str,
    expert_count_key: &'static str,
    nested: Option<(&'static str, &'static str)>,
}

const FAMILIES: [FamilyLayout; 2] = [
    FamilyLayout {
        gguf_architecture: "granitemoe",
        model_type: "granitemoe",
        architecture_class: "GraniteMoeForCausalLM",
        expert_count_key: "num_local_experts",
        nested: None,
    },
    FamilyLayout {
        gguf_architecture: "gemma4",
        model_type: "gemma4",
        architecture_class: "Gemma4ForConditionalGeneration",
        expert_count_key: "num_experts",
        nested: Some(("text_config", "gemma4_text")),
    },
];

/// Writes the `config.json` bytes for an MLX export of `hparams`.
///
/// `quantization` is `Some((bits, group_size))` on an affine-quantized export,
/// which adds `quantization` and its `quantization_config` mirror with
/// `mode: "affine"`; a plain export (`None`) carries neither key.
///
/// # Errors
///
/// [`InteropError::UnknownFamilyConfig`] when `hparams.family` has no layout
/// here; [`InteropError::MalformedHfConfig`] when a float field is not finite or
/// the per-layer KV heads differ (HF has one `num_key_value_heads`).
#[must_use = "the config bytes are the export's config.json"]
pub fn hf_config_bytes(
    hparams: &ModelHparams,
    quantization: Option<(u8, u32)>,
) -> Result<Vec<u8>, InteropError> {
    let layout = family_layout(&hparams.family)?;
    let mut fields = text_fields(hparams, layout)?;
    let mut root = fields.clone();
    insert_field(&mut root, "model_type", json!(layout.model_type))?;
    insert_field(
        &mut root,
        "architectures",
        json!([layout.architecture_class]),
    )?;
    if let Some((name, nested_type)) = layout.nested {
        insert_field(&mut fields, "model_type", json!(nested_type))?;
        root.insert(name.into(), Value::Object(fields));
    }
    if let Some((bits, group_size)) = quantization {
        let block = json!({"group_size": group_size, "bits": bits, "mode": "affine"});
        root.insert("quantization".into(), block.clone());
        root.insert("quantization_config".into(), block);
    }
    serde_json::to_vec_pretty(&Value::Object(root)).map_err(|error| malformed(&error.to_string()))
}

/// Carries an HF-directory source's `config.json` into the export unchanged.
/// The bytes are checked to be a JSON object, nothing more: a multimodal
/// source nests its text fields, so [`crate::hf_config::HfConfig`] is not the
/// right validator.
///
/// # Errors
///
/// [`InteropError::MalformedHfConfig`] when `source` is not a JSON object.
#[must_use = "the config bytes are the export's config.json"]
pub fn copy_config(source: &[u8]) -> Result<Vec<u8>, InteropError> {
    match serde_json::from_slice::<Value>(source) {
        Ok(Value::Object(_)) => Ok(source.to_vec()),
        Ok(_) => Err(malformed("config.json is not a json object")),
        Err(error) => Err(malformed(&error.to_string())),
    }
}

fn family_layout(family: &str) -> Result<&'static FamilyLayout, InteropError> {
    FAMILIES
        .iter()
        .find(|layout| layout.gguf_architecture == family)
        .ok_or_else(|| InteropError::UnknownFamilyConfig {
            family: family.to_string(),
        })
}

fn text_fields(
    hparams: &ModelHparams,
    layout: &FamilyLayout,
) -> Result<Map<String, Value>, InteropError> {
    let mut pairs = alloc::vec![
        ("hidden_size", json!(hparams.embedding)),
        ("num_hidden_layers", json!(hparams.block_count)),
        ("num_attention_heads", json!(hparams.query_heads)),
        ("num_key_value_heads", json!(uniform_kv_heads(hparams)?)),
        ("intermediate_size", json!(hparams.feed_forward)),
        ("vocab_size", json!(hparams.vocab)),
        ("head_dim", json!(hparams.head_dim)),
        ("rms_norm_eps", float_value(hparams.rms_epsilon)?),
        ("rope_theta", float_value(hparams.rope_freq_base)?),
        ("tie_word_embeddings", json!(hparams.tied_embeddings)),
    ];
    if hparams.expert_count > 0 {
        pairs.push((layout.expert_count_key, json!(hparams.expert_count)));
        pairs.push(("num_experts_per_tok", json!(hparams.expert_used_count)));
    }
    let mut fields = Map::new();
    pairs
        .into_iter()
        .try_for_each(|(key, value)| insert_field(&mut fields, key, value))?;
    Ok(fields)
}

fn insert_field(
    fields: &mut Map<String, Value>,
    key: &str,
    value: Value,
) -> Result<(), InteropError> {
    if !HF_CONFIG_KEYS.contains(&key) {
        return Err(malformed(&alloc::format!(
            "key {key} is not one the reader consults"
        )));
    }
    fields.insert(key.into(), value);
    Ok(())
}

fn uniform_kv_heads(hparams: &ModelHparams) -> Result<u32, InteropError> {
    let first = hparams.kv_heads_by_layer.first().copied().unwrap_or(0);
    match hparams
        .kv_heads_by_layer
        .iter()
        .all(|heads| *heads == first)
    {
        true => Ok(first),
        false => Err(malformed(
            "per-layer kv heads differ; hf has one num_key_value_heads",
        )),
    }
}

fn float_value(value: f32) -> Result<Value, InteropError> {
    let shortest = value
        .to_string()
        .parse::<f64>()
        .map_err(|error| malformed(&error.to_string()))?;
    Number::from_f64(shortest)
        .map(Value::Number)
        .ok_or_else(|| malformed("float field is not finite"))
}

fn malformed(reason: &str) -> InteropError {
    InteropError::MalformedHfConfig {
        reason: reason.to_string(),
    }
}

/// Reads the dimensions an MLX config needs from a sliding-pattern GGUF
/// (Gemma 4), whose per-layer `feed_forward_length` array
/// [`crate::bind::architecture_from_metadata`] refuses. `head_dim` is the
/// sliding layers' `key_length_swa`, which is what HF's `head_dim` names (the
/// full layers' width is `global_head_dim`); `feed_forward` is the first
/// layer's width, the base `use_double_wide_mlp` widens.
#[must_use]
pub fn sliding_pattern_hparams(hparams: &SlidingPatternHparams) -> ModelHparams {
    let kv_heads = hparams.kv_heads_by_layer.first().copied().unwrap_or(0);
    ModelHparams {
        vocab: hparams.vocab,
        embedding: hparams.embedding,
        feed_forward: hparams.feed_forward,
        query_heads: hparams.head_count,
        kv_heads,
        kv_heads_by_layer: hparams.kv_heads_by_layer.clone(),
        head_dim: hparams.key_length_swa,
        block_count: hparams.block_count,
        expert_count: hparams.expert_count,
        expert_used_count: hparams.expert_used_count,
        rope_freq_base: hparams.rope_freq_base,
        rms_epsilon: hparams.rms_epsilon,
        tied_embeddings: false,
        family: String::from("gemma4"),
        sliding_rope: None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::fs::File;
    use std::path::Path;

    use proxima_gguf::parse_complete;

    use super::*;
    use crate::bind::architecture_from_metadata;
    use crate::hf_config::{architecture_from_hf_config, parse_hf_config};
    use crate::sliding_pattern::hparams::from_metadata as sliding_from_metadata;

    const E2B_GGUF: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
    const GRANITE_GGUF: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";
    const SPEC_KEYS: [&str; 10] = [
        "architectures",
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "rms_norm_eps",
        "rope_theta",
    ];

    fn with_parsed<T>(path: &str, read: impl FnOnce(&proxima_gguf::pipe::ParsedGguf) -> T) -> T {
        let file = File::open(path).expect("host-local gguf blob present");
        let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("gguf maps");
        read(&parse_complete(&mapping).expect("gguf parses"))
    }

    fn granite_hparams() -> ModelHparams {
        with_parsed(GRANITE_GGUF, |parsed| {
            architecture_from_metadata(parsed).expect("granite metadata reads")
        })
    }

    fn e2b_hparams() -> ModelHparams {
        with_parsed(E2B_GGUF, |parsed| {
            sliding_pattern_hparams(&sliding_from_metadata(parsed).expect("e2b metadata reads"))
        })
    }

    fn vendored(model: &str) -> (Vec<u8>, Value) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/mlx-reference")
            .join(model)
            .join("plain/config.json");
        let bytes = std::fs::read(path).expect("vendored config.json present");
        let value = serde_json::from_slice(&bytes).expect("vendored config.json parses");
        (bytes, value)
    }

    fn written(hparams: &ModelHparams, quantization: Option<(u8, u32)>) -> Value {
        let bytes = hf_config_bytes(hparams, quantization).expect("known family writes");
        serde_json::from_slice(&bytes).expect("written config parses")
    }

    fn read_back(config: &Value) -> ModelHparams {
        let bytes = serde_json::to_vec(config).expect("config serializes");
        architecture_from_hf_config(&parse_hf_config(&bytes).expect("reader accepts config"))
            .expect("reader derives hparams")
    }

    fn differing_keys(left: &Value, right: &Value) -> Vec<String> {
        let (left, right) = (left.as_object().unwrap(), right.as_object().unwrap());
        left.keys()
            .chain(right.keys())
            .filter(|key| left.get(*key) != right.get(*key))
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn first_difference(left: &[u8], right: &[u8]) -> Option<usize> {
        left.iter()
            .zip(right)
            .position(|(a_byte, b_byte)| a_byte != b_byte)
            .or((left.len() != right.len()).then_some(left.len().min(right.len())))
    }

    fn oracle_has(vendored: &Value, key: &str) -> bool {
        let text = vendored.get("text_config").unwrap_or(vendored);
        text.get(key).is_some()
            || vendored.get(key).is_some()
            || (key == "rope_theta" && text.get("rope_parameters").is_some())
    }

    fn assert_spec_keys_equal_vendored(config: &Value, vendored: &Value, hparams: &ModelHparams) {
        for key in SPEC_KEYS {
            assert!(config.get(key).is_some(), "written config lacks {key}");
            assert!(oracle_has(vendored, key), "vendored config lacks {key}");
        }
        assert_eq!(config["hidden_size"], json!(hparams.embedding));
        assert_eq!(config["num_hidden_layers"], json!(hparams.block_count));
        assert_eq!(config["num_attention_heads"], json!(hparams.query_heads));
        assert_eq!(config["num_key_value_heads"], json!(hparams.kv_heads));
        assert_eq!(config["intermediate_size"], json!(hparams.feed_forward));
        assert_eq!(config["vocab_size"], json!(hparams.vocab));
        assert_eq!(config["model_type"], vendored["model_type"]);
        assert_eq!(config["architectures"], vendored["architectures"]);
    }

    #[test]
    fn mlx_config_granite_carries_every_key_and_moe_field_with_the_gguf_values() {
        let hparams = granite_hparams();
        let (_, vendored) = vendored("granite-3.1-1b-a400m-moe");

        let config = written(&hparams, None);

        assert_spec_keys_equal_vendored(&config, &vendored, &hparams);
        for key in [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
            "rms_norm_eps",
            "rope_theta",
            "num_local_experts",
            "num_experts_per_tok",
        ] {
            assert_eq!(
                config[key], vendored[key],
                "granite {key} differs from the vendored config"
            );
        }
    }

    #[test]
    fn mlx_config_e2b_carries_every_key_with_the_gguf_values_and_no_moe_fields() {
        let hparams = e2b_hparams();
        let (_, vendored) = vendored("gemma4-e2b");

        let config = written(&hparams, None);

        assert_spec_keys_equal_vendored(&config, &vendored, &hparams);
        let nested = &config["text_config"];
        let oracle = &vendored["text_config"];
        for key in [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
            "rms_norm_eps",
            "head_dim",
        ] {
            assert_eq!(
                nested[key], oracle[key],
                "e2b text_config {key} differs from the vendored config"
            );
        }
        assert_eq!(
            config["rope_theta"],
            oracle["rope_parameters"]["full_attention"]["rope_theta"]
        );
        assert!(config.get("num_experts").is_none() && config.get("num_local_experts").is_none());
    }

    #[test]
    fn mlx_config_identity_control_same_hparams_write_the_same_config() {
        let hparams = granite_hparams();

        let differing = differing_keys(&written(&hparams, None), &written(&hparams, None));

        assert_eq!(differing, Vec::<String>::new());
    }

    #[test]
    fn mlx_config_degenerate_control_one_altered_value_is_the_only_difference() {
        let hparams = granite_hparams();
        let altered = ModelHparams {
            embedding: hparams.embedding + 1,
            ..hparams.clone()
        };

        let differing = differing_keys(&written(&hparams, None), &written(&altered, None));

        assert_eq!(differing, vec!["hidden_size".to_string()]);
    }

    #[test]
    fn mlx_config_granite_read_back_equals_the_gguf_hparams() {
        let hparams = granite_hparams();

        assert_eq!(read_back(&written(&hparams, None)), hparams);
    }

    #[test]
    fn mlx_config_e2b_read_back_equals_the_gguf_hparams() {
        let hparams = e2b_hparams();

        assert_eq!(read_back(&written(&hparams, None)), hparams);
    }

    #[test]
    fn mlx_config_degenerate_control_altered_hidden_size_reads_back_different() {
        let hparams = granite_hparams();
        let mut config = written(&hparams, None);
        config["hidden_size"] = json!(hparams.embedding + 1);

        assert_ne!(read_back(&config), hparams);
    }

    #[test]
    fn mlx_config_unknown_family_is_refused_by_name() {
        let hparams = ModelHparams {
            family: "llama".into(),
            ..granite_hparams()
        };

        let error = hf_config_bytes(&hparams, None).expect_err("llama has no layout");

        assert!(matches!(error, InteropError::UnknownFamilyConfig { family } if family == "llama"));
    }

    #[test]
    fn mlx_config_quantized_export_carries_quantization_and_its_mirror() {
        let config = written(&granite_hparams(), Some((4, 64)));

        let expected = json!({"group_size": 64, "bits": 4, "mode": "affine"});
        assert_eq!(config["quantization"], expected);
        assert_eq!(config["quantization_config"], expected);
    }

    #[test]
    fn mlx_config_plain_export_carries_neither_quantization_key() {
        let config = written(&granite_hparams(), None);

        assert!(config.get("quantization").is_none());
        assert!(config.get("quantization_config").is_none());
    }

    #[test]
    fn mlx_config_copy_of_an_hf_directory_source_is_byte_equal() {
        for model in ["granite-3.1-1b-a400m-moe", "gemma4-e2b"] {
            let (source, _) = vendored(model);

            let copied = copy_config(&source).expect("vendored config is a json object");

            assert_eq!(first_difference(&copied, &source), None, "{model}");
        }
    }

    #[test]
    fn mlx_config_copy_identity_control_equal_buffers_have_no_difference() {
        let (source, _) = vendored("granite-3.1-1b-a400m-moe");

        assert_eq!(first_difference(&source, &source.clone()), None);
    }

    #[test]
    fn mlx_config_copy_degenerate_control_one_byte_different_source_is_located() {
        let (source, _) = vendored("granite-3.1-1b-a400m-moe");
        let position = source
            .iter()
            .position(|byte| *byte == b'1')
            .expect("config has a digit");
        let mut altered = source.clone();
        altered[position] = b'2';

        let copied = copy_config(&altered).expect("altered config is still a json object");

        assert_eq!(first_difference(&copied, &source), Some(position));
    }
}
