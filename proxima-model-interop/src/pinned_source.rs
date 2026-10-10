use alloc::collections::BTreeMap;
use alloc::string::String;

use proxima_safetensors::{Manifest, ShardedIndex, parse_complete, parse_sharded_index};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPin {
    pub repository: String,
    pub revision: String,
    /// relative checkpoint artifact paths and their exact byte hashes
    pub artifacts: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PinnedCheckpoint<'source> {
    pub provenance: CheckpointPin,
    pub config: serde_json::Value,
    pub index: ShardedIndex,
    pub shards: BTreeMap<String, PinnedShard<'source>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PinnedShard<'source> {
    pub manifest: Manifest,
    pub bytes: &'source [u8],
}

impl PinnedShard<'_> {
    #[must_use]
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let tensor = self.manifest.tensor(name)?;
        let header_length = u64::from_le_bytes(self.bytes.get(..8)?.try_into().ok()?);
        let data_start = usize::try_from(header_length).ok()?.checked_add(8)?;
        let start = data_start.checked_add(usize::try_from(tensor.data_offsets.0).ok()?)?;
        let end = data_start.checked_add(usize::try_from(tensor.data_offsets.1).ok()?)?;
        self.bytes.get(start..end)
    }
}

#[derive(Debug, Error)]
pub enum PinnedSourceError {
    #[error("checkpoint repository or revision differs from the pin")]
    RevisionMismatch,
    #[error("checkpoint revision must be a lowercase 40-character commit hash")]
    InvalidRevision,
    #[error("checkpoint artifact {path:?} has no sha256 pin")]
    MissingHash { path: String },
    #[error("checkpoint artifact {path:?} has an invalid sha256 pin")]
    InvalidHash { path: String },
    #[error("checkpoint artifact {path:?} is absent")]
    MissingArtifact { path: String },
    #[error("checkpoint artifact {path:?} differs from its sha256 pin")]
    HashMismatch { path: String },
    #[error("checkpoint artifact path {path:?} must be relative without parent components")]
    InvalidPath { path: String },
    #[error("checkpoint tensor {tensor:?} disagrees with its shard index")]
    TensorIndexMismatch { tensor: String },
    #[error("checkpoint tensor index is empty")]
    EmptyIndex,
    #[error(transparent)]
    Config(#[from] serde_json::Error),
    #[error(transparent)]
    Safetensors(#[from] proxima_safetensors::SafetensorsError),
}

#[must_use = "checkpoint provenance must be checked before binding"]
pub fn read_pinned_checkpoint<'source>(
    pin: &CheckpointPin,
    repository: &str,
    revision: &str,
    artifacts: &BTreeMap<String, &'source [u8]>,
) -> Result<PinnedCheckpoint<'source>, PinnedSourceError> {
    if pin.repository != repository || pin.revision != revision {
        return Err(PinnedSourceError::RevisionMismatch);
    }
    if !lowercase_hex(revision, 40) {
        return Err(PinnedSourceError::InvalidRevision);
    }
    let config = serde_json::from_slice(verified_artifact(pin, artifacts, "config.json")?)?;
    let index = parse_sharded_index(verified_artifact(
        pin,
        artifacts,
        "model.safetensors.index.json",
    )?)?;
    if index.weight_map.is_empty() {
        return Err(PinnedSourceError::EmptyIndex);
    }
    let mut shards = BTreeMap::new();
    for filename in index.shard_filenames() {
        let bytes = verified_artifact(pin, artifacts, filename)?;
        let manifest = parse_complete(bytes)?;
        for tensor in &manifest.tensors {
            if index.shard_for(&tensor.name) != Some(filename) {
                return Err(PinnedSourceError::TensorIndexMismatch {
                    tensor: tensor.name.clone(),
                });
            }
        }
        for tensor in index.tensors_in_shard(filename) {
            if manifest.tensor(tensor).is_none() {
                return Err(PinnedSourceError::TensorIndexMismatch {
                    tensor: String::from(tensor),
                });
            }
        }
        shards.insert(String::from(filename), PinnedShard { manifest, bytes });
    }
    for path in pin.artifacts.keys().filter(|path| {
        path.as_str() != "config.json"
            && path.as_str() != "model.safetensors.index.json"
            && !shards.contains_key(*path)
    }) {
        verified_artifact(pin, artifacts, path)?;
    }
    Ok(PinnedCheckpoint {
        provenance: pin.clone(),
        config,
        index,
        shards,
    })
}

fn verified_artifact<'bytes>(
    pin: &CheckpointPin,
    artifacts: &BTreeMap<String, &'bytes [u8]>,
    path: &str,
) -> Result<&'bytes [u8], PinnedSourceError> {
    if path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path
            .split('/')
            .any(|component| matches!(component, "" | "." | ".."))
    {
        return Err(PinnedSourceError::InvalidPath { path: path.into() });
    }
    let expected = pin
        .artifacts
        .get(path)
        .ok_or_else(|| PinnedSourceError::MissingHash { path: path.into() })?;
    if !lowercase_hex(expected, 64) {
        return Err(PinnedSourceError::InvalidHash { path: path.into() });
    }
    let bytes = artifacts
        .get(path)
        .ok_or_else(|| PinnedSourceError::MissingArtifact { path: path.into() })?;
    let actual = alloc::format!("{:x}", Sha256::digest(bytes));
    if actual != *expected {
        return Err(PinnedSourceError::HashMismatch { path: path.into() });
    }
    Ok(bytes)
}

fn lowercase_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use std::fs;

    use sha2::{Digest, Sha256};
    use tempfile::tempdir;

    use super::{CheckpointPin, PinnedSourceError, read_pinned_checkpoint};

    #[test]
    fn architecture_matrix_pinned_source() {
        let inventory: serde_json::Value = serde_json::from_str(
            include_str!(
                "../../proxima-tensor/specs/small-model-architecture-matrix/inventory.jsonl"
            )
            .lines()
            .next()
            .expect("inventory contains a pinned candidate"),
        )
        .expect("parse inventory record");
        let repository = inventory["repo"].as_str().expect("repository pin");
        let revision = inventory["revision"].as_str().expect("revision pin");
        let directory = tempdir().expect("create checkpoint fixture store");
        let header =
            br#"{"model.embed_tokens.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}}"#;
        let mut weights = (header.len() as u64).to_le_bytes().to_vec();
        weights.extend_from_slice(header);
        weights.extend_from_slice(&1.0_f32.to_le_bytes());
        weights.extend_from_slice(&(-2.0_f32).to_le_bytes());
        let payloads = [
            ("config.json", br#"{"model_type":"lfm2","fixture":true}"#.as_slice()),
            ("model.safetensors.index.json", br#"{"metadata":{"total_size":8},"weight_map":{"model.embed_tokens.weight":"model-00001-of-00001.safetensors"}}"#.as_slice()),
            ("model-00001-of-00001.safetensors", weights.as_slice()),
        ];
        let pin = CheckpointPin {
            repository: repository.into(),
            revision: revision.into(),
            artifacts: payloads
                .iter()
                .map(|(path, bytes)| {
                    (
                        String::from(*path),
                        alloc::format!("{:x}", Sha256::digest(bytes)),
                    )
                })
                .collect(),
        };
        for (path, bytes) in payloads {
            fs::write(directory.path().join(path), bytes)
                .expect("write checkpoint fixture artifact");
        }
        fs::write(
            directory.path().join("source.json"),
            serde_json::to_vec(&pin).expect("serialize provenance"),
        )
        .expect("write checkpoint provenance");
        let on_disk_pin: CheckpointPin = serde_json::from_slice(
            &fs::read(directory.path().join("source.json")).expect("read provenance"),
        )
        .expect("parse provenance");
        let stored: BTreeMap<String, _> = on_disk_pin
            .artifacts
            .keys()
            .map(|path| {
                (
                    path.clone(),
                    fs::read(directory.path().join(path)).expect("read pinned artifact"),
                )
            })
            .collect();
        let views = stored
            .iter()
            .map(|(path, bytes)| (path.clone(), bytes.as_slice()))
            .collect();
        let accepted = read_pinned_checkpoint(&on_disk_pin, repository, revision, &views)
            .expect("read exact pinned fixture source");
        assert_eq!(accepted.provenance, pin);
        assert_eq!(accepted.config["model_type"], "lfm2");
        assert_eq!(
            accepted.index.shard_for("model.embed_tokens.weight"),
            Some("model-00001-of-00001.safetensors")
        );
        assert_eq!(
            accepted.shards["model-00001-of-00001.safetensors"]
                .manifest
                .tensors[0]
                .shape,
            [1, 2]
        );
        let expected_tensor_bytes = [1.0_f32.to_le_bytes(), (-2.0_f32).to_le_bytes()].concat();
        assert_eq!(
            accepted.shards["model-00001-of-00001.safetensors"]
                .tensor_bytes("model.embed_tokens.weight"),
            Some(expected_tensor_bytes.as_slice())
        );
        assert!(matches!(
            read_pinned_checkpoint(
                &pin,
                repository,
                "0000000000000000000000000000000000000000",
                &views
            ),
            Err(PinnedSourceError::RevisionMismatch)
        ));
        let mut missing_hash = pin.clone();
        missing_hash
            .artifacts
            .remove("model-00001-of-00001.safetensors");
        assert!(
            matches!(read_pinned_checkpoint(&missing_hash, repository, revision, &views), Err(PinnedSourceError::MissingHash { path }) if path == "model-00001-of-00001.safetensors")
        );
        let mut wrong_hash = pin.clone();
        wrong_hash
            .artifacts
            .insert("config.json".into(), "0".repeat(64));
        assert!(
            matches!(read_pinned_checkpoint(&wrong_hash, repository, revision, &views), Err(PinnedSourceError::HashMismatch { path }) if path == "config.json")
        );
        std::println!(
            "accepted=1 wrong_revision_rejected=1 missing_hash_rejected=1 tensor_payload_bytes=8"
        );
    }
}
