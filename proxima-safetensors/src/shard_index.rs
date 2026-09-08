//! `model.safetensors.index.json` -- the sharded-checkpoint manifest.
//!
//! Not a [`crate::parser::SafetensorsParser`] concern: that FSM's whole
//! contract is "one safetensors wire buffer in, one [`crate::parser::Manifest`]
//! out" (`lib.rs`'s own module doc). The index file is a different, much
//! smaller JSON document -- `{"metadata": {"total_size": u64}, "weight_map":
//! {tensor_name: shard_filename}}` -- that a caller reads BEFORE it knows
//! which shard file to hand to the parser at all. So this is its own sans-IO
//! value type with a pure `parse` function, not a new state machine: the
//! whole document arrives in one buffer (index files are kilobytes, never
//! chunked), so there is no accumulation to do a Pipe or an FSM would earn
//! its keep over.
//!
//! Also not a [`proxima_primitives::pipe::Pipe`]: parsing one complete JSON
//! buffer into a `BTreeMap` is a pure fold with no I/O boundary of its own --
//! the same "identical call site" test the local-quant-plan's own "what is
//! NOT a pipe" section applies to this exact type.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::error::SafetensorsError;

/// One shard's worth of tensor names, resolved from `weight_map`. Callers
/// (the quantizer's shard source) group by shard file so each shard's mmap
/// is opened exactly once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedIndex {
    /// Total declared byte size across every shard (`metadata.total_size`),
    /// `None` if the index omitted it.
    pub total_size: Option<u64>,
    /// Tensor name -> the shard filename that holds it, in the order the
    /// JSON object's keys were declared (`BTreeMap` sorts by name, which is
    /// stable and sufficient -- the shard source drives tensors by name
    /// order, not file-declaration order, matching `Manifest::tensor`'s own
    /// by-name lookup contract).
    pub weight_map: BTreeMap<String, String>,
}

impl ShardedIndex {
    /// Every tensor name mapped to shard `shard_filename`, in `weight_map`'s
    /// (sorted, stable) name order.
    pub fn tensors_in_shard<'index>(
        &'index self,
        shard_filename: &'index str,
    ) -> impl Iterator<Item = &'index str> {
        self.weight_map
            .iter()
            .filter(move |(_name, shard)| shard.as_str() == shard_filename)
            .map(|(name, _shard)| name.as_str())
    }

    /// Every distinct shard filename referenced by `weight_map`, in sorted
    /// order -- the shard source's own open-order.
    #[must_use]
    pub fn shard_filenames(&self) -> Vec<&str> {
        let mut filenames: Vec<&str> = self
            .weight_map
            .values()
            .map(String::as_str)
            .collect::<alloc::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        filenames.sort_unstable();
        filenames
    }

    /// The shard filename holding `tensor_name`, if the index knows it.
    #[must_use]
    pub fn shard_for(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(String::as_str)
    }
}

/// Parses a `model.safetensors.index.json` buffer. Sans-IO: the caller
/// already read the whole file into `bytes` (index files are small, no
/// chunked FSM needed -- see the module doc).
///
/// # Errors
///
/// [`SafetensorsError::MalformedJson`] if `bytes` isn't valid JSON;
/// [`SafetensorsError::HeaderNotAnObject`] if the top level or `weight_map`
/// isn't a JSON object; [`SafetensorsError::MissingField`] if `weight_map`
/// is absent; [`SafetensorsError::InvalidField`] if a `weight_map` value
/// isn't a string.
pub fn parse_sharded_index(bytes: &[u8]) -> Result<ShardedIndex, SafetensorsError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| SafetensorsError::MalformedJson {
            reason: alloc::string::ToString::to_string(&error),
        })?;
    let object = value
        .as_object()
        .ok_or(SafetensorsError::HeaderNotAnObject)?;

    let total_size = object
        .get("metadata")
        .and_then(serde_json::Value::as_object)
        .and_then(|metadata| metadata.get("total_size"))
        .and_then(serde_json::Value::as_u64);

    let weight_map_value =
        object
            .get("weight_map")
            .ok_or_else(|| SafetensorsError::MissingField {
                tensor: "__index__".into(),
                field: "weight_map",
            })?;
    let weight_map_object = weight_map_value
        .as_object()
        .ok_or(SafetensorsError::HeaderNotAnObject)?;

    let mut weight_map = BTreeMap::new();
    for (name, shard_value) in weight_map_object {
        let shard = shard_value
            .as_str()
            .ok_or_else(|| SafetensorsError::InvalidField {
                tensor: name.clone(),
                field: "weight_map",
            })?;
        weight_map.insert(name.clone(), String::from(shard));
    }

    Ok(ShardedIndex {
        total_size,
        weight_map,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[proxima::test]
    async fn parses_weight_map_and_total_size() {
        let json = br#"{
            "metadata": {"total_size": 4070000000},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors"
            }
        }"#;

        let index = parse_sharded_index(json).expect("well-formed index parses");

        assert_eq!(index.total_size, Some(4_070_000_000));
        assert_eq!(index.weight_map.len(), 3);
        assert_eq!(
            index.shard_for("lm_head.weight"),
            Some("model-00002-of-00002.safetensors")
        );
        assert_eq!(index.shard_for("no.such.tensor"), None);
    }

    #[proxima::test]
    async fn groups_tensor_names_by_shard_file() {
        let json = br#"{
            "weight_map": {
                "a": "shard-1.safetensors",
                "b": "shard-1.safetensors",
                "c": "shard-2.safetensors"
            }
        }"#;
        let index = parse_sharded_index(json).expect("well-formed index parses");

        assert_eq!(index.shard_filenames(), vec!["shard-1.safetensors", "shard-2.safetensors"]);

        let mut shard_one: Vec<&str> = index.tensors_in_shard("shard-1.safetensors").collect();
        shard_one.sort_unstable();
        assert_eq!(shard_one, vec!["a", "b"]);

        let shard_two: Vec<&str> = index.tensors_in_shard("shard-2.safetensors").collect();
        assert_eq!(shard_two, vec!["c"]);
    }

    #[proxima::test]
    async fn missing_total_size_is_none_not_an_error() {
        let json = br#"{"weight_map": {"a": "shard-1.safetensors"}}"#;
        let index = parse_sharded_index(json).expect("total_size is optional");
        assert_eq!(index.total_size, None);
    }

    #[proxima::test]
    async fn missing_weight_map_is_a_typed_error() {
        let json = br#"{"metadata": {"total_size": 1}}"#;
        let outcome = parse_sharded_index(json);
        assert!(matches!(
            outcome,
            Err(SafetensorsError::MissingField { field: "weight_map", .. })
        ));
    }

    #[proxima::test]
    async fn non_string_shard_value_is_a_typed_error() {
        let json = br#"{"weight_map": {"a": 123}}"#;
        let outcome = parse_sharded_index(json);
        assert!(matches!(
            outcome,
            Err(SafetensorsError::InvalidField { field: "weight_map", .. })
        ));
    }

    #[proxima::test]
    async fn malformed_json_is_a_typed_error() {
        let outcome = parse_sharded_index(b"not json");
        assert!(matches!(outcome, Err(SafetensorsError::MalformedJson { .. })));
    }
}
