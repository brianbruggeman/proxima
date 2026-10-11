//! MLX shard assembly: the byte buffers and file names of an mlx-lm model
//! directory's weight files, with no file ever opened.
//!
//! Composes [`proxima_safetensors::write_complete`] for each shard and
//! [`proxima_safetensors::parse_sharded_index`]'s document shape for the
//! index. The cut mirrors mlx-lm's `make_shards`: tensors stay in order and a
//! new shard starts when the next tensor would push the running total past the
//! cap. A caller that owns the filesystem writes `shards` and `index` out; this
//! module is the sans-IO half so the same assembly serves any edge.
//!
//! `write_complete` always writes its own `proxima_format_version` key into
//! `__metadata__`; `format` is a different key, so both coexist in each shard.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_safetensors::{SafetensorsError, SafetensorsModel, TensorPayload, write_complete};

use crate::error::InteropError;

/// mlx-lm `utils.py` `MAX_FILE_SIZE_GB = 5`, shifted by `<< 30`.
pub const MLX_MAX_SHARD_BYTES: u64 = 5 << 30;

pub const MLX_SINGLE_SHARD_FILE_NAME: &str = "model.safetensors";

pub const MLX_INDEX_FILE_NAME: &str = "model.safetensors.index.json";

const MLX_FORMAT_KEY: &str = "format";

const MLX_FORMAT_VALUE: &str = "mlx";

/// One weight file: its name in the model directory and its full bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxShard {
    pub file_name: String,
    pub bytes: Vec<u8>,
}

/// The weight files of one MLX model directory. `index` holds the bytes of
/// [`MLX_INDEX_FILE_NAME`] and is `None` for a single shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxExport {
    pub shards: Vec<MlxShard>,
    pub index: Option<Vec<u8>>,
}

impl MlxExport {
    /// Cuts `tensors` into shards of at most `cap_bytes` payload bytes each
    /// (a single tensor larger than the cap gets a shard of its own, as in
    /// mlx-lm) and renders every shard plus the index when there are several.
    /// Production callers pass [`MLX_MAX_SHARD_BYTES`].
    ///
    /// # Errors
    ///
    /// [`InteropError::Safetensors`] when `write_complete` rejects a tensor
    /// (duplicate name, payload length not matching shape and dtype).
    pub fn assemble(tensors: Vec<TensorPayload<'_>>, cap_bytes: u64) -> Result<Self, InteropError> {
        let total_size: u64 = tensors.iter().map(payload_bytes).sum();
        let groups = cut_shards(tensors, cap_bytes);
        let shard_count = groups.len();
        let mut weight_map = BTreeMap::new();
        let mut shards = Vec::with_capacity(shard_count);
        for (position, group) in groups.into_iter().enumerate() {
            let file_name = shard_file_name(position, shard_count);
            weight_map.extend(
                group
                    .iter()
                    .map(|tensor| (tensor.name.clone(), file_name.clone())),
            );
            shards.push(render_shard(group, file_name)?);
        }
        let index = (shard_count > 1)
            .then(|| render_index(total_size, &weight_map))
            .transpose()?;
        Ok(Self { shards, index })
    }
}

fn payload_bytes(tensor: &TensorPayload<'_>) -> u64 {
    tensor.data.len() as u64
}

fn cut_shards<'a>(tensors: Vec<TensorPayload<'a>>, cap_bytes: u64) -> Vec<Vec<TensorPayload<'a>>> {
    let mut groups: Vec<Vec<TensorPayload<'a>>> = Vec::new();
    let mut running = 0u64;
    for tensor in tensors {
        let size = payload_bytes(&tensor);
        match groups.last_mut() {
            Some(group) if running.saturating_add(size) <= cap_bytes => {
                running += size;
                group.push(tensor);
            }
            _ => {
                running = size;
                groups.push(alloc::vec![tensor]);
            }
        }
    }
    if groups.is_empty() {
        groups.push(Vec::new());
    }
    groups
}

fn shard_file_name(position: usize, shard_count: usize) -> String {
    match shard_count {
        1 => MLX_SINGLE_SHARD_FILE_NAME.to_string(),
        _ => format!(
            "model-{:05}-of-{:05}.safetensors",
            position + 1,
            shard_count
        ),
    }
}

fn render_shard(
    tensors: Vec<TensorPayload<'_>>,
    file_name: String,
) -> Result<MlxShard, InteropError> {
    let metadata = BTreeMap::from([(MLX_FORMAT_KEY.to_string(), MLX_FORMAT_VALUE.to_string())]);
    let bytes = write_complete(&SafetensorsModel { tensors, metadata })?;
    Ok(MlxShard { file_name, bytes })
}

fn render_index(
    total_size: u64,
    weight_map: &BTreeMap<String, String>,
) -> Result<Vec<u8>, InteropError> {
    let document = serde_json::json!({
        "metadata": { "total_size": total_size },
        "weight_map": weight_map,
    });
    serde_json::to_vec(&document).map_err(|error| {
        InteropError::Safetensors(SafetensorsError::MalformedJson {
            reason: error.to_string(),
        })
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use proxima_safetensors::{parse_complete, parse_sharded_index};
    use proxima_tensor::DType;

    use super::*;

    const HIDDEN: u64 = 8;
    const TENSOR_BYTES: u64 = HIDDEN * HIDDEN * 2;

    fn weight_bytes(seed: usize) -> Vec<u8> {
        (0..TENSOR_BYTES as usize)
            .map(|index| ((index * 31 + seed * 7) % 251) as u8)
            .collect()
    }

    fn layer_names(count: usize) -> Vec<String> {
        (0..count)
            .map(|layer| format!("model.layers.{layer}.self_attn.q_proj.weight"))
            .collect()
    }

    fn payloads<'a>(names: &[String], buffers: &'a [Vec<u8>]) -> Vec<TensorPayload<'a>> {
        names
            .iter()
            .zip(buffers)
            .map(|(name, data)| TensorPayload {
                name: name.clone(),
                dtype: DType::BFloat16,
                shape: alloc::vec![HIDDEN, HIDDEN],
                data,
            })
            .collect()
    }

    fn assemble_layers(count: usize, cap_bytes: u64) -> (Vec<String>, MlxExport) {
        let names = layer_names(count);
        let buffers: Vec<Vec<u8>> = (0..count).map(weight_bytes).collect();
        let export = MlxExport::assemble(payloads(&names, &buffers), cap_bytes)
            .expect("assembles layer shards");
        (names, export)
    }

    fn shard_holds(shard: &MlxShard, name: &str) -> bool {
        parse_complete(&shard.bytes)
            .expect("parses shard")
            .tensor(name)
            .is_some()
    }

    fn shard_format(shard: &MlxShard) -> Option<String> {
        let manifest = parse_complete(&shard.bytes).expect("parses shard");
        manifest.metadata.get("format").cloned()
    }

    fn two_shard_cap() -> u64 {
        TENSOR_BYTES * 2
    }

    #[test]
    fn mlx_shard_single_shard_metadata_declares_format_mlx() {
        let (_, export) = assemble_layers(2, MLX_MAX_SHARD_BYTES);

        assert_eq!(shard_format(&export.shards[0]).as_deref(), Some("mlx"));
    }

    #[test]
    fn mlx_shard_each_of_two_shards_declares_format_mlx() {
        let (_, export) = assemble_layers(4, two_shard_cap());

        let formats: Vec<_> = export.shards.iter().map(shard_format).collect();

        assert_eq!(formats, alloc::vec![Some("mlx".to_string()); 2]);
    }

    #[test]
    fn mlx_shard_cap_constant_equals_mlx_lm_five_gib() {
        assert_eq!(MLX_MAX_SHARD_BYTES, 5_368_709_120);
        assert_eq!(MLX_MAX_SHARD_BYTES, 5u64 << 30);
    }

    #[test]
    fn mlx_shard_cap_constant_is_not_five_decimal_gigabytes() {
        assert_ne!(MLX_MAX_SHARD_BYTES, 5_000_000_000);
    }

    #[test]
    fn mlx_shard_model_under_injected_cap_yields_one_shard() {
        let (_, export) = assemble_layers(3, TENSOR_BYTES * 3);

        assert_eq!(export.shards.len(), 1);
    }

    #[test]
    fn mlx_shard_model_over_injected_cap_yields_two_shards() {
        let (_, export) = assemble_layers(4, two_shard_cap());

        assert_eq!(export.shards.len(), 2);
    }

    #[test]
    fn mlx_shard_single_shard_is_named_model_safetensors() {
        let (_, export) = assemble_layers(2, MLX_MAX_SHARD_BYTES);

        assert_eq!(export.shards[0].file_name, "model.safetensors");
    }

    #[test]
    fn mlx_shard_multi_shard_names_carry_position_and_count() {
        let (_, export) = assemble_layers(4, two_shard_cap());

        let names: Vec<_> = export
            .shards
            .iter()
            .map(|shard| shard.file_name.as_str())
            .collect();

        assert_eq!(
            names,
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        );
    }

    #[test]
    fn mlx_shard_each_tensor_lands_in_exactly_one_shard() {
        let (names, export) = assemble_layers(3, two_shard_cap());

        let holders: Vec<usize> = names
            .iter()
            .map(|name| {
                export
                    .shards
                    .iter()
                    .filter(|shard| shard_holds(shard, name))
                    .count()
            })
            .collect();

        assert_eq!(holders, alloc::vec![1; 3]);
    }

    #[test]
    fn mlx_shard_index_weight_map_names_the_holding_shard_for_every_tensor() {
        let (names, export) = assemble_layers(3, two_shard_cap());

        let index = parse_sharded_index(&export.index.expect("two shards emit an index"))
            .expect("parses index");

        for name in &names {
            let holder = export.shards.iter().find(|shard| shard_holds(shard, name));
            assert_eq!(
                index.shard_for(name),
                holder.map(|shard| shard.file_name.as_str())
            );
        }
    }

    #[test]
    fn mlx_shard_index_total_size_equals_summed_payload_bytes() {
        let (names, export) = assemble_layers(3, two_shard_cap());

        let index = parse_sharded_index(&export.index.expect("two shards emit an index"))
            .expect("parses index");

        assert_eq!(index.total_size, Some(TENSOR_BYTES * names.len() as u64));
    }

    #[test]
    fn mlx_shard_single_shard_emits_no_index_file() {
        let (_, export) = assemble_layers(2, MLX_MAX_SHARD_BYTES);

        assert_eq!(export.index, None);
    }
}
