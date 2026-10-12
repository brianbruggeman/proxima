use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io;
use std::path::Path;
use std::process::Command;

use memmap2::{Mmap, MmapOptions};
use omega::backend;
use proxima_model_interop::{HfConfig, parse_hf_config};
use proxima_primitives::Codec;
use proxima_safetensors::Manifest;
use proxima_tensor::QuantizedBlock;
use proxima_tokenizer::Vocab;
use serde_json::{Value, json};

use super::graph::Shape;

pub const REVISION: &str = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1";
pub const DIGEST: &str = "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af";
pub const NORM_NAMES: [&str; 4] = [
    "input_layernorm.weight",
    "input_layernorm_2.weight",
    "post_attention_layernorm.weight",
    "post_attention_layernorm_2.weight",
];
pub const MATRIX_NAMES: [&str; 7] = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum WeightMappingMode {
    WholeCheckpoint,
    PackedCopies,
    CpuUnmapped,
}

impl WeightMappingMode {
    pub fn for_run(backend_name: &str, metal_copy_weights: bool) -> Self {
        match (backend_name, metal_copy_weights) {
            ("metal", false) => Self::WholeCheckpoint,
            ("metal", true) => Self::PackedCopies,
            _ => Self::CpuUnmapped,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::WholeCheckpoint => "whole_checkpoint_mapping",
            Self::PackedCopies => "packed_weight_copies",
            Self::CpuUnmapped => "cpu_unmapped",
        }
    }
}

pub struct Entry {
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

pub struct Checkpoint {
    mapping: Mmap,
    pub manifest: Manifest,
    pub hf_config: HfConfig,
    pub data_start: u64,
    pub mapping_mode: WeightMappingMode,
    mapping_registered: bool,
    pub entries: BTreeMap<String, Entry>,
    pub layer_scales: Vec<Vec<Vec<f32>>>,
    pub final_scale: Vec<f32>,
    pub shape: Shape,
    pub vocab: Vocab,
    pub inventory: Vec<Value>,
}

fn number(config: &Value, name: &str) -> io::Result<u32> {
    let value = config
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| io::Error::other(format!("missing unsigned config field {name}")))?;
    u32::try_from(value).map_err(io::Error::other)
}

fn decode(bytes: &[u8]) -> io::Result<Vec<f32>> {
    let mut values = vec![0.0; bytes.len() / 2];
    proxima_gguf::quant::bf16::dequantize(bytes, &mut values).map_err(io::Error::other)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(io::Error::other(
            "checkpoint scale or embedding has nonfinite values",
        ));
    }
    Ok(values)
}

impl Checkpoint {
    pub fn open(directory: &Path, mapping_mode: WeightMappingMode) -> io::Result<Self> {
        let config_bytes = fs::read(directory.join("config.json"))?;
        let hf_config = parse_hf_config(&config_bytes).map_err(io::Error::other)?;
        let config: Value = serde_json::from_slice(&config_bytes).map_err(io::Error::other)?;
        let shape = Shape {
            hidden: number(&config, "hidden_size")?,
            intermediate: number(&config, "intermediate_size")?,
            heads: number(&config, "num_attention_heads")?,
            head_width: number(&config, "head_dim")?,
            vocab: number(&config, "vocab_size")?,
        };
        if config["model_type"] != "ouro"
            || shape.hidden != 2048
            || shape.intermediate != 5632
            || shape.heads != 16
            || shape.head_width != 128
            || shape.vocab != 49152
            || number(&config, "num_key_value_heads")? != 16
            || number(&config, "num_hidden_layers")? != 24
            || number(&config, "total_ut_steps")? != 4
            || config["hidden_act"] != "silu"
            || config["tie_word_embeddings"] != false
            || !config["sliding_window"].is_null()
            || !config["rope_scaling"].is_null()
        {
            return Err(io::Error::other(
                "config does not match the admitted pinned Ouro architecture",
            ));
        }
        let types = config["layer_types"]
            .as_array()
            .ok_or_else(|| io::Error::other("layer_types missing"))?;
        if types.len() != 24 || types.iter().any(|value| value != "full_attention") {
            return Err(io::Error::other("Ouro requires 24 full attention layers"));
        }
        let epsilon = config["rms_norm_eps"]
            .as_f64()
            .ok_or_else(|| io::Error::other("rms_norm_eps missing"))? as f32;
        let rope_base = config["rope_theta"]
            .as_f64()
            .ok_or_else(|| io::Error::other("rope_theta missing"))? as f32;
        if epsilon != 0.000001 || rope_base != 1000000.0 {
            return Err(io::Error::other(
                "normalization or RoPE config differs from pinned checkpoint",
            ));
        }
        let weight_path = directory.join("model.safetensors");
        let digest_output = Command::new("/usr/bin/openssl")
            .args(["dgst", "-sha256"])
            .arg(&weight_path)
            .output()?;
        if !digest_output.status.success() {
            return Err(io::Error::other(format!(
                "checkpoint digest utility failed: {}",
                String::from_utf8_lossy(&digest_output.stderr)
            )));
        }
        let digest_text = String::from_utf8(digest_output.stdout).map_err(io::Error::other)?;
        let observed_digest = digest_text
            .split_whitespace()
            .last()
            .ok_or_else(|| io::Error::other("checkpoint digest absent"))?;
        if observed_digest != DIGEST {
            return Err(io::Error::other(format!(
                "checkpoint digest differs: observed {observed_digest}, required {DIGEST}"
            )));
        }
        let file = File::open(weight_path)?;
        // the mapping owner remains alive until all borrowed execution buffers are dropped.
        let mapping = unsafe { MmapOptions::new().map(&file)? };
        let manifest = proxima_safetensors::parse_complete(&mapping).map_err(io::Error::other)?;
        let first: [u8; 8] = mapping
            .get(..8)
            .ok_or_else(|| io::Error::other("safetensors length word absent"))?
            .try_into()
            .map_err(io::Error::other)?;
        let header_length = usize::try_from(u64::from_le_bytes(first)).map_err(io::Error::other)?;
        let data_start = 8usize
            .checked_add(header_length)
            .ok_or_else(|| io::Error::other("header length overflow"))?;
        let header_bytes = mapping
            .get(8..data_start)
            .ok_or_else(|| io::Error::other("header exceeds mapped file"))?;
        let header: Value = serde_json::from_slice(header_bytes).map_err(io::Error::other)?;
        let object = header
            .as_object()
            .ok_or_else(|| io::Error::other("safetensors header is not an object"))?;
        let mut entries = BTreeMap::new();
        let mut inventory = Vec::new();
        for (name, record) in object {
            if name == "__metadata__" {
                continue;
            }
            if record["dtype"] != "BF16" {
                return Err(io::Error::other(format!("{name}: expected BF16")));
            }
            let dimensions = record["shape"]
                .as_array()
                .ok_or_else(|| io::Error::other(format!("{name}: shape absent")))?;
            let dimensions: Vec<usize> = dimensions
                .iter()
                .map(|dimension| {
                    dimension
                        .as_u64()
                        .ok_or_else(|| io::Error::other("invalid tensor dimension"))
                        .and_then(|dimension| usize::try_from(dimension).map_err(io::Error::other))
                })
                .collect::<io::Result<_>>()?;
            let count = dimensions.iter().try_fold(1usize, |count, dimension| {
                count
                    .checked_mul(*dimension)
                    .ok_or_else(|| io::Error::other("tensor shape overflow"))
            })?;
            let offsets = record["data_offsets"]
                .as_array()
                .ok_or_else(|| io::Error::other("data_offsets absent"))?;
            if offsets.len() != 2 {
                return Err(io::Error::other("data_offsets must have two entries"));
            }
            let begin = usize::try_from(
                offsets[0]
                    .as_u64()
                    .ok_or_else(|| io::Error::other("invalid begin offset"))?,
            )
            .map_err(io::Error::other)?;
            let finish = usize::try_from(
                offsets[1]
                    .as_u64()
                    .ok_or_else(|| io::Error::other("invalid end offset"))?,
            )
            .map_err(io::Error::other)?;
            if finish.checked_sub(begin) != count.checked_mul(2) {
                return Err(io::Error::other(format!(
                    "{name}: BF16 byte count differs from shape"
                )));
            }
            let start = data_start
                .checked_add(begin)
                .ok_or_else(|| io::Error::other("data begin overflow"))?;
            let end = data_start
                .checked_add(finish)
                .ok_or_else(|| io::Error::other("data end overflow"))?;
            if mapping.get(start..end).is_none() {
                return Err(io::Error::other(format!(
                    "{name}: byte range exceeds checkpoint"
                )));
            }
            inventory.push(json!({"kind":"tensor", "name":name,"dtype":"BF16","shape":dimensions,"data_offsets":[begin,finish],"absolute_offsets":[start,end],"bytes":end-start}));
            entries.insert(
                name.clone(),
                Entry {
                    shape: dimensions,
                    start,
                    end,
                },
            );
        }
        let mut ordered: Vec<&Entry> = entries.values().collect();
        ordered.sort_by_key(|entry| entry.start);
        if ordered.windows(2).any(|pair| pair[0].end > pair[1].start) {
            return Err(io::Error::other("checkpoint tensor ranges overlap"));
        }
        let tokenizer: Value = serde_json::from_slice(&fs::read(directory.join("tokenizer.json"))?)
            .map_err(io::Error::other)?;
        let vocabulary = tokenizer["model"]["vocab"]
            .as_object()
            .ok_or_else(|| io::Error::other("tokenizer vocabulary absent"))?;
        let mut tokens = vec![String::new(); shape.vocab as usize];
        for (token, identifier) in vocabulary {
            let identifier = usize::try_from(
                identifier
                    .as_u64()
                    .ok_or_else(|| io::Error::other("invalid token id"))?,
            )
            .map_err(io::Error::other)?;
            let slot = tokens
                .get_mut(identifier)
                .ok_or_else(|| io::Error::other("token id exceeds configured vocabulary"))?;
            if !slot.is_empty() {
                return Err(io::Error::other("duplicate token id"));
            }
            *slot = token.clone();
        }
        if tokens.iter().any(String::is_empty) {
            return Err(io::Error::other("incomplete tokenizer vocabulary"));
        }
        let pairs = tokenizer["model"]["merges"]
            .as_array()
            .ok_or_else(|| io::Error::other("tokenizer merges absent"))?;
        let mut merges = Vec::with_capacity(pairs.len());
        for pair in pairs {
            let pair = pair
                .as_array()
                .filter(|pair| pair.len() == 2)
                .ok_or_else(|| io::Error::other("expected BPE merge pair"))?;
            let left = pair[0]
                .as_str()
                .ok_or_else(|| io::Error::other("merge left absent"))?;
            let right = pair[1]
                .as_str()
                .ok_or_else(|| io::Error::other("merge right absent"))?;
            merges.push(format!("{left} {right}"));
        }
        let vocab =
            Vocab::new(tokens, &merges, Some(0), Some(0), None).map_err(io::Error::other)?;
        let mut checkpoint = Self {
            mapping,
            manifest,
            hf_config,
            data_start: u64::try_from(data_start).map_err(io::Error::other)?,
            mapping_mode,
            mapping_registered: false,
            entries,
            layer_scales: Vec::new(),
            final_scale: Vec::new(),
            shape,
            vocab,
            inventory,
        };
        let mut validated_layer_tensors = 0;
        for layer in 0..24 {
            let mut scales = Vec::new();
            for suffix in NORM_NAMES {
                let name = format!("model.layers.{layer}.{suffix}");
                checkpoint.require_shape(&name, &[2048])?;
                validated_layer_tensors += 1;
                scales.push(decode(checkpoint.bytes(&name)?)?);
            }
            for (index, suffix) in MATRIX_NAMES.iter().enumerate() {
                let expected = match index {
                    4 | 5 => [5632, 2048],
                    6 => [2048, 5632],
                    _ => [2048, 2048],
                };
                checkpoint.require_shape(&format!("model.layers.{layer}.{suffix}"), &expected)?;
                validated_layer_tensors += 1;
            }
            checkpoint.layer_scales.push(scales);
        }
        let mut validated_shared_tensors = 0;
        for (name, shape) in [
            ("model.embed_tokens.weight", vec![49152, 2048]),
            ("model.norm.weight", vec![2048]),
            ("lm_head.weight", vec![49152, 2048]),
        ] {
            checkpoint.require_shape(name, &shape)?;
            validated_shared_tensors += 1;
        }
        let mut validated_gate_tensors = 0;
        for (name, shape) in [
            ("model.early_exit_gate.weight", vec![1, 2048]),
            ("model.early_exit_gate.bias", vec![1]),
        ] {
            checkpoint.require_shape(name, &shape)?;
            validated_gate_tensors += 1;
        }
        checkpoint.final_scale = decode(checkpoint.bytes("model.norm.weight")?)?;
        checkpoint.inventory.push(json!({"kind":"inventory_counts","layer_tensor_records":validated_layer_tensors,"shared_model_tensor_records":validated_shared_tensors,"gate_selection_tensor_records":validated_gate_tensors,"actual_header_tensor_records":checkpoint.entries.len(),"range_shape_dtype_violations":0,"checkpoint_bytes":checkpoint.mapping.len(),"revision":REVISION,"expected_sha256":DIGEST,"observed_sha256":observed_digest,"digest_verification_source":"native /usr/bin/openssl dgst -sha256 on mapped checkpoint path"}));
        if checkpoint.mapping_mode == WeightMappingMode::WholeCheckpoint {
            backend::register_checkpoint_mapping(&checkpoint.mapping);
            checkpoint.mapping_registered = true;
        }
        Ok(checkpoint)
    }

    fn require_shape(&self, name: &str, expected: &[usize]) -> io::Result<()> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| io::Error::other(format!("tensor {name} absent")))?;
        if entry.shape != expected {
            return Err(io::Error::other(format!(
                "{name}: observed shape {:?}, required {expected:?}",
                entry.shape
            )));
        }
        Ok(())
    }

    pub fn bytes(&self, name: &str) -> io::Result<&[u8]> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| io::Error::other(format!("tensor {name} absent")))?;
        self.mapping
            .get(entry.start..entry.end)
            .ok_or_else(|| io::Error::other("mapped tensor range absent"))
    }

    pub fn mapped_bytes(&self) -> &[u8] {
        &self.mapping
    }

    pub fn packed(&self, name: &str) -> io::Result<QuantizedBlock<'_>> {
        Ok(QuantizedBlock::Packed {
            codec: Codec::BFloat16,
            bytes: self.bytes(name)?,
        })
    }

    pub fn embedding_rows(&self, prefix: &[u32]) -> io::Result<Vec<f32>> {
        let bytes = self.bytes("model.embed_tokens.weight")?;
        let stride = self.shape.hidden as usize * 2;
        let mut values = Vec::with_capacity(prefix.len() * self.shape.hidden as usize);
        for token in prefix {
            if *token >= self.shape.vocab {
                return Err(io::Error::other("prefix token outside vocabulary"));
            }
            let start = *token as usize * stride;
            values.extend(decode(&bytes[start..start + stride])?);
        }
        Ok(values)
    }
}

impl Drop for Checkpoint {
    fn drop(&mut self) {
        if self.mapping_registered {
            backend::unregister_checkpoint_mapping(&self.mapping);
        }
    }
}
