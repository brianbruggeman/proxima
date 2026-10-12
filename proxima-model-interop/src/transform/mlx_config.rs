//! HF `config.json` writer for the MLX export, the inverse of
//! [`crate::hf_config::parse_hf_config`] and [`crate::hf_config::architecture_from_hf_config`].
//!
//! Sans-IO: [`hf_config_bytes`] takes the [`ModelHparams`] that
//! [`crate::bind::architecture_from_metadata`] read from the GGUF plus the
//! [`ParsedGguf`] itself, and returns the file's bytes; the edge writes them.
//! [`copy_config`] is the HF-directory counterpart, which carries the source
//! file through untouched. Every key written is drawn from [`HF_CONFIG_KEYS`]
//! (the list the reader consults) or [`HF_CONFIG_PASSTHROUGH_KEYS`] (what the
//! reader ignores but mlx-lm reads), so a field renamed on one side is a
//! compile or test failure on the other. A key with no GGUF source and no
//! family constant is not written; `mlx_config_fields` lists them.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use serde_json::{Map, Number, Value, json};

use super::mlx_config_fields::{
    FamilyFields, GEMMA_FLOATS, GEMMA_INTS, GRANITE_FLOATS, GRANITE_INTS, KeyTable, gemma_fields,
    granite_fields, table_fields, tied_embeddings,
};
use super::mlx_conventions::QuantizedConventions;
use super::mlx_names::{Layout, SANDWICH_NORM_LAYOUT, STANDARD_LAYOUT};
use crate::bind::{ModelHparams, architecture_from_metadata, metadata_str};
use crate::error::InteropError;
use crate::hf_config::{HF_CONFIG_KEYS, HF_CONFIG_PASSTHROUGH_KEYS};
use crate::sliding_pattern::hparams::{SlidingPatternHparams, from_metadata};

/// How one GGUF `general.architecture` spells itself in an HF `config.json`.
/// `nested` names the sub-object (and its `model_type`) a multimodal
/// checkpoint keeps its text fields under; only `model_type`, `architectures`
/// and `tie_word_embeddings` stay at the root, as the reference does, and
/// `parse_hf_config` reads through the nesting. `floats` and `ints` map HF keys
/// to `{architecture}.` GGUF suffixes; `computed` derives the rest.
struct FamilyLayout {
    gguf_architecture: &'static str,
    model_type: &'static str,
    architecture_class: &'static str,
    expert_count_key: &'static str,
    nested: Option<(&'static str, &'static str)>,
    floats: KeyTable,
    ints: KeyTable,
    computed: FamilyFields,
    tensor_layout: Layout,
    tensor_prefix: &'static str,
    quantized: QuantizedConventions,
    hparams: HparamsReader,
}

type HparamsReader = fn(&ParsedGguf) -> Result<ModelHparams, InteropError>;

type FamilyExport = (Layout, &'static str, QuantizedConventions, ModelHparams);

/// mlx-lm `granitemoe.py`: `sanitize` splits `input_linear` into
/// `switch_mlp.gate_proj` and `switch_mlp.up_proj` and renames `output_linear`
/// to `switch_mlp.down_proj` before it quantizes, and `quant_predicate` gives
/// `router.layer` 8 bits.
pub(super) const GRANITE_QUANTIZED: QuantizedConventions = QuantizedConventions {
    renames: &[
        (
            "block_sparse_moe.input_linear",
            &[
                "block_sparse_moe.switch_mlp.gate_proj",
                "block_sparse_moe.switch_mlp.up_proj",
            ],
        ),
        (
            "block_sparse_moe.output_linear",
            &["block_sparse_moe.switch_mlp.down_proj"],
        ),
    ],
    bits: &[("block_sparse_moe.router.layer", 8)],
};

const FAMILIES: [FamilyLayout; 2] = [
    FamilyLayout {
        gguf_architecture: "granitemoe",
        model_type: "granitemoe",
        architecture_class: "GraniteMoeForCausalLM",
        expert_count_key: "num_local_experts",
        nested: None,
        floats: GRANITE_FLOATS,
        ints: GRANITE_INTS,
        computed: granite_fields,
        tensor_layout: STANDARD_LAYOUT,
        tensor_prefix: "model.",
        quantized: GRANITE_QUANTIZED,
        hparams: architecture_from_metadata,
    },
    FamilyLayout {
        gguf_architecture: "gemma4",
        model_type: "gemma4",
        architecture_class: "Gemma4ForConditionalGeneration",
        expert_count_key: "num_experts",
        nested: Some(("text_config", "gemma4_text")),
        floats: GEMMA_FLOATS,
        ints: GEMMA_INTS,
        computed: gemma_fields,
        tensor_layout: SANDWICH_NORM_LAYOUT,
        tensor_prefix: "language_model.model.",
        quantized: QuantizedConventions::NONE,
        hparams: gemma_hparams,
    },
];

/// Writes the `config.json` bytes for an MLX export of `hparams`, with every
/// extra key the GGUF `metadata` (or its tensor directory) can supply.
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
    metadata: &ParsedGguf,
    quantization: Option<(u8, u32)>,
) -> Result<Vec<u8>, InteropError> {
    hf_config_bytes_with_overrides(hparams, metadata, quantization, &[])
}

/// [`hf_config_bytes`] for an export whose modules are not all quantized at
/// the same width: each `(module path, bits)` of `overrides` adds a
/// `quantization.<module path>` entry at the export's group size, which is the
/// per-module form mlx-lm's loader reads.
///
/// # Errors
///
/// As [`hf_config_bytes`].
#[must_use = "the config bytes are the export's config.json"]
pub fn hf_config_bytes_with_overrides(
    hparams: &ModelHparams,
    metadata: &ParsedGguf,
    quantization: Option<(u8, u32)>,
    overrides: &[(String, u8)],
) -> Result<Vec<u8>, InteropError> {
    let layout = family_layout(&hparams.family)?;
    let fields = text_fields(hparams, metadata, layout)?;
    let mut root = root_fields(&fields, layout)?;
    if let Some((name, nested_type)) = layout.nested {
        let mut nested = fields;
        insert_field(&mut nested, "model_type", json!(nested_type))?;
        root.insert(name.into(), Value::Object(nested));
    }
    if let Some((bits, group_size)) = quantization {
        let mut block = json!({"group_size": group_size, "bits": bits, "mode": "affine"});
        if let Some(entries) = block.as_object_mut() {
            overrides.iter().for_each(|(module, module_bits)| {
                entries.insert(
                    module.clone(),
                    json!({"group_size": group_size, "bits": module_bits}),
                );
            });
        }
        root.insert("quantization".into(), block.clone());
        root.insert("quantization_config".into(), block);
    }
    serde_json::to_vec_pretty(&Value::Object(root)).map_err(|error| malformed(&error.to_string()))
}

fn root_fields(
    fields: &Map<String, Value>,
    layout: &FamilyLayout,
) -> Result<Map<String, Value>, InteropError> {
    let mut root = match layout.nested {
        None => fields.clone(),
        Some(_) => fields
            .get("tie_word_embeddings")
            .map(|tied| Map::from_iter([("tie_word_embeddings".to_string(), tied.clone())]))
            .unwrap_or_default(),
    };
    insert_field(&mut root, "model_type", json!(layout.model_type))?;
    insert_field(
        &mut root,
        "architectures",
        json!([layout.architecture_class]),
    )?;
    Ok(root)
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

/// The tensor naming and hyperparameters an MLX export of `parsed` needs:
/// the name layout, the HF prefix of the decoder tensors, the
/// [`QuantizedConventions`] of the family and the [`ModelHparams`] its reader
/// builds from the GGUF.
///
/// # Errors
///
/// [`InteropError::UnknownFamilyConfig`] for a family with no layout here;
/// the metadata errors of the family's reader.
pub(super) fn family_export(parsed: &ParsedGguf) -> Result<FamilyExport, InteropError> {
    let layout = family_layout(metadata_str(parsed, "general.architecture")?)?;
    let hparams = (layout.hparams)(parsed)?;
    Ok((
        layout.tensor_layout,
        layout.tensor_prefix,
        layout.quantized,
        hparams,
    ))
}

fn gemma_hparams(parsed: &ParsedGguf) -> Result<ModelHparams, InteropError> {
    Ok(sliding_pattern_hparams(&from_metadata(parsed)?))
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
    metadata: &ParsedGguf,
    layout: &FamilyLayout,
) -> Result<Map<String, Value>, InteropError> {
    let mut pairs = dimension_pairs(hparams, metadata, layout)?;
    pairs.extend(table_fields(
        metadata,
        layout.gguf_architecture,
        layout.floats,
        layout.ints,
    )?);
    pairs.extend((layout.computed)(metadata, hparams)?);
    let mut fields = Map::new();
    pairs
        .into_iter()
        .try_for_each(|(key, value)| insert_field(&mut fields, key, value))?;
    Ok(fields)
}

fn dimension_pairs(
    hparams: &ModelHparams,
    metadata: &ParsedGguf,
    layout: &FamilyLayout,
) -> Result<Vec<(&'static str, Value)>, InteropError> {
    let mut pairs = alloc::vec![
        ("hidden_size", json!(hparams.embedding)),
        ("num_hidden_layers", json!(hparams.block_count)),
        ("num_attention_heads", json!(hparams.query_heads)),
        ("num_key_value_heads", json!(uniform_kv_heads(hparams)?)),
        ("intermediate_size", json!(hparams.feed_forward)),
        ("vocab_size", json!(hparams.vocab)),
        ("rms_norm_eps", float_value(hparams.rms_epsilon)?),
        (
            "tie_word_embeddings",
            json!(tied_embeddings(hparams, metadata))
        ),
    ];
    if layout.nested.is_none() {
        pairs.push(("rope_theta", float_value(hparams.rope_freq_base)?));
    }
    if hparams.head_dim != hparams.embedding / hparams.query_heads.max(1) {
        pairs.push(("head_dim", json!(hparams.head_dim)));
    }
    if hparams.expert_count > 0 {
        pairs.push((layout.expert_count_key, json!(hparams.expert_count)));
        pairs.push(("num_experts_per_tok", json!(hparams.expert_used_count)));
    }
    Ok(pairs)
}

fn insert_field(
    fields: &mut Map<String, Value>,
    key: &str,
    value: Value,
) -> Result<(), InteropError> {
    if !HF_CONFIG_KEYS.contains(&key) && !HF_CONFIG_PASSTHROUGH_KEYS.contains(&key) {
        return Err(malformed(&alloc::format!(
            "key {key} is not one the reader consults or the writer passes through"
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

pub(super) fn float_value(value: f32) -> Result<Value, InteropError> {
    let shortest = value
        .to_string()
        .parse::<f64>()
        .map_err(|error| malformed(&error.to_string()))?;
    Number::from_f64(shortest)
        .map(Value::Number)
        .ok_or_else(|| malformed("float field is not finite"))
}

pub(super) fn malformed(reason: &str) -> InteropError {
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
    use proxima_gguf::pipe::ParsedGguf;

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
    const GRANITE_DERIVED_KEYS: [&str; 12] = [
        "attention_bias",
        "attention_multiplier",
        "bos_token_id",
        "embedding_multiplier",
        "eos_token_id",
        "hidden_act",
        "logits_scaling",
        "max_position_embeddings",
        "pad_token_id",
        "residual_multiplier",
        "tie_word_embeddings",
        "vocab_size",
    ];
    const E2B_TEXT_DERIVED_KEYS: [&str; 17] = [
        "bos_token_id",
        "enable_moe_block",
        "eos_token_id",
        "final_logit_softcapping",
        "global_head_dim",
        "head_dim",
        "hidden_activation",
        "hidden_size_per_layer_input",
        "layer_types",
        "max_position_embeddings",
        "num_kv_shared_layers",
        "pad_token_id",
        "rope_parameters",
        "sliding_window",
        "use_double_wide_mlp",
        "vocab_size_per_layer_input",
        "vocab_size",
    ];

    fn open(path: &str) -> ParsedGguf {
        let file = File::open(path).expect("host-local gguf blob present");
        let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("gguf maps");
        parse_complete(&mapping).expect("gguf parses")
    }

    fn granite() -> (ModelHparams, ParsedGguf) {
        let parsed = open(GRANITE_GGUF);
        (
            with_directory_tie(
                architecture_from_metadata(&parsed).expect("granite metadata reads"),
                &parsed,
            ),
            parsed,
        )
    }

    fn e2b() -> (ModelHparams, ParsedGguf) {
        let parsed = open(E2B_GGUF);
        let hparams =
            sliding_pattern_hparams(&sliding_from_metadata(&parsed).expect("e2b metadata reads"));
        (with_directory_tie(hparams, &parsed), parsed)
    }

    fn with_directory_tie(hparams: ModelHparams, parsed: &ParsedGguf) -> ModelHparams {
        ModelHparams {
            tied_embeddings: tied_embeddings(&hparams, parsed),
            ..hparams
        }
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

    fn written(loaded: &(ModelHparams, ParsedGguf), quantization: Option<(u8, u32)>) -> Value {
        let bytes =
            hf_config_bytes(&loaded.0, &loaded.1, quantization).expect("known family writes");
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

    fn written_has(config: &Value, key: &str) -> bool {
        let text = config.get("text_config").unwrap_or(config);
        text.get(key).is_some() || (key == "rope_theta" && text.get("rope_parameters").is_some())
    }

    fn assert_spec_keys_equal_vendored(config: &Value, vendored: &Value, hparams: &ModelHparams) {
        let text = config.get("text_config").unwrap_or(config);
        for key in SPEC_KEYS {
            assert!(
                written_has(config, key) || config.get(key).is_some(),
                "written config lacks {key}"
            );
            assert!(oracle_has(vendored, key), "vendored config lacks {key}");
        }
        assert_eq!(text["hidden_size"], json!(hparams.embedding));
        assert_eq!(text["num_hidden_layers"], json!(hparams.block_count));
        assert_eq!(text["num_attention_heads"], json!(hparams.query_heads));
        assert_eq!(text["num_key_value_heads"], json!(hparams.kv_heads));
        assert_eq!(text["intermediate_size"], json!(hparams.feed_forward));
        assert_eq!(text["vocab_size"], json!(hparams.vocab));
        assert_eq!(config["model_type"], vendored["model_type"]);
        assert_eq!(config["architectures"], vendored["architectures"]);
    }

    fn assert_keys_equal(written: &Value, oracle: &Value, keys: &[&str], label: &str) {
        for key in keys {
            assert_eq!(
                written[*key], oracle[*key],
                "{label} {key} differs from the vendored config"
            );
        }
    }

    #[test]
    fn mlx_config_granite_carries_every_key_and_moe_field_with_the_gguf_values() {
        let loaded = granite();
        let (_, vendored) = vendored("granite-3.1-1b-a400m-moe");

        let config = written(&loaded, None);

        assert_spec_keys_equal_vendored(&config, &vendored, &loaded.0);
        let keys = [
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
        ];
        assert_keys_equal(&config, &vendored, &keys, "granite");
    }

    #[test]
    fn mlx_config_granite_carries_the_multipliers_token_ids_and_context_the_gguf_names() {
        let loaded = granite();
        let (_, vendored) = vendored("granite-3.1-1b-a400m-moe");

        let config = written(&loaded, None);

        assert_keys_equal(&config, &vendored, &GRANITE_DERIVED_KEYS, "granite");
        assert_eq!(config["residual_multiplier"], json!(0.22));
        assert_eq!(config["max_position_embeddings"], json!(131_072));
    }

    #[test]
    fn mlx_config_granite_writes_no_head_dim_and_none_of_the_underivable_keys() {
        let config = written(&granite(), None);

        let keys: Vec<&str> = config
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();

        for absent in [
            "head_dim",
            "torch_dtype",
            "transformers_version",
            "initializer_range",
            "attention_dropout",
            "use_cache",
            "output_router_logits",
            "router_aux_loss_coef",
            "rope_scaling",
        ] {
            assert!(!keys.contains(&absent), "granite config carries {absent}");
        }
    }

    #[test]
    fn mlx_config_e2b_carries_every_key_with_the_gguf_values_and_no_moe_fields() {
        let loaded = e2b();
        let (_, vendored) = vendored("gemma4-e2b");

        let config = written(&loaded, None);

        assert_spec_keys_equal_vendored(&config, &vendored, &loaded.0);
        let keys = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
            "rms_norm_eps",
            "head_dim",
        ];
        assert_keys_equal(
            &config["text_config"],
            &vendored["text_config"],
            &keys,
            "e2b",
        );
        let nested = &config["text_config"];
        assert_eq!(
            nested["rope_parameters"]["full_attention"]["rope_theta"],
            vendored["text_config"]["rope_parameters"]["full_attention"]["rope_theta"]
        );
        assert!(nested.get("num_experts").is_none() && nested.get("num_local_experts").is_none());
    }

    #[test]
    fn mlx_config_e2b_nests_the_text_keys_and_writes_no_tower_blocks() {
        let config = written(&e2b(), None);

        let root: Vec<&str> = config
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();

        assert_eq!(
            root,
            vec![
                "architectures",
                "model_type",
                "text_config",
                "tie_word_embeddings"
            ]
        );
        assert_eq!(config["text_config"]["model_type"], json!("gemma4_text"));
    }

    #[test]
    fn mlx_config_e2b_text_config_carries_the_gguf_derived_values_the_reference_names() {
        let loaded = e2b();
        let (_, vendored) = vendored("gemma4-e2b");

        let config = written(&loaded, None);

        let derived = &E2B_TEXT_DERIVED_KEYS[..];
        let comparable: Vec<&str> = derived
            .iter()
            .copied()
            .filter(|key| *key != "rope_parameters" && *key != "vocab_size_per_layer_input")
            .collect();
        assert_keys_equal(
            &config["text_config"],
            &vendored["text_config"],
            &comparable,
            "e2b",
        );
        let sliding = &config["text_config"]["rope_parameters"]["sliding_attention"];
        assert_eq!(
            sliding["rope_theta"],
            vendored["text_config"]["rope_parameters"]["sliding_attention"]["rope_theta"]
        );
        assert_eq!(
            config["text_config"]["vocab_size_per_layer_input"],
            vendored["text_config"]["vocab_size_per_layer_input"]
        );
    }

    #[test]
    fn mlx_config_identity_control_same_hparams_write_the_same_config() {
        let loaded = granite();

        let differing = differing_keys(&written(&loaded, None), &written(&loaded, None));

        assert_eq!(differing, Vec::<String>::new());
    }

    #[test]
    fn mlx_config_degenerate_control_one_altered_value_is_the_only_difference() {
        let (hparams, parsed) = granite();
        let altered = ModelHparams {
            embedding: hparams.embedding + 1,
            ..hparams.clone()
        };

        let differing = differing_keys(
            &written(&(hparams, parsed.clone()), None),
            &written(&(altered, parsed), None),
        );

        assert_eq!(differing, vec!["hidden_size".to_string()]);
    }

    #[test]
    fn mlx_config_granite_read_back_equals_the_gguf_hparams() {
        let loaded = granite();

        assert_eq!(read_back(&written(&loaded, None)), loaded.0);
    }

    #[test]
    fn mlx_config_e2b_read_back_equals_the_gguf_hparams() {
        let loaded = e2b();

        assert_eq!(read_back(&written(&loaded, None)), loaded.0);
    }

    #[test]
    fn mlx_config_degenerate_control_altered_hidden_size_reads_back_different() {
        let loaded = granite();
        let mut config = written(&loaded, None);
        config["hidden_size"] = json!(loaded.0.embedding + 1);

        assert_ne!(read_back(&config), loaded.0);
    }

    #[test]
    fn mlx_config_unknown_family_is_refused_by_name() {
        let (hparams, parsed) = granite();
        let unknown = ModelHparams {
            family: "llama".into(),
            ..hparams
        };

        let error = hf_config_bytes(&unknown, &parsed, None).expect_err("llama has no layout");

        assert!(matches!(error, InteropError::UnknownFamilyConfig { family } if family == "llama"));
    }

    #[test]
    fn mlx_config_quantized_export_carries_quantization_and_its_mirror() {
        let config = written(&granite(), Some((4, 64)));

        let expected = json!({"group_size": 64, "bits": 4, "mode": "affine"});
        assert_eq!(config["quantization"], expected);
        assert_eq!(config["quantization_config"], expected);
    }

    #[test]
    fn mlx_config_plain_export_carries_neither_quantization_key() {
        let config = written(&granite(), None);

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
