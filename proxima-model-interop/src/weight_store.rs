//! A directory of recoded, page-aligned checkpoints, keyed by what produced them.
//!
//! The shared safetensors binder recodes some leaves on every first bind
//! (`bind_materialized`); [`crate::ModelWeights::persist_aligned`] writes the
//! result back out as a checkpoint the next process maps instead of the source.
//! This module is only the keyed directory: [`store_key`] names an entry,
//! [`lookup`] finds a published one, [`publish`] writes one atomically. It
//! carries no model knowledge, so any host that binds through the shared binder
//! composes it the same way.
//!
//! The key is a plain `u128` (an `xxh3` digest), not a newtype: a caller's
//! `store_path(dir, key)` reads identically with or without a wrapper, so a
//! wrapper would only relocate the line.
//!
//! # Staleness
//!
//! The key covers the open source file's device, inode, length, mtime and
//! ctime (nanoseconds, one `fstat` on the handle that is mapped), its header
//! bytes, the rules, the alignment and [`STORE_LAYOUT_VERSION`]. It does not
//! cover tensor bytes. Two distinct files never share a key, and an in-place
//! edit changes ctime. A copy to a new inode, or a device renumbered after a
//! reboot, is a new key and costs one cold recode. Only a rewrite that also
//! forges ctime serves a stale entry.

use std::fs::{self, File};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

use xxhash_rust::xxh3::Xxh3;

use crate::error::InteropError;
use crate::serving::{NamePattern, WeightPrecisionRule};

#[cfg(feature = "instrument")]
use proxima_telemetry::debug;

/// Bumped whenever the written layout changes in a way an old store cannot
/// satisfy, so every earlier entry stops matching.
pub const STORE_LAYOUT_VERSION: u32 = 1;

const STORE_EXTENSION: &str = "safetensors";
const TORN_EXTENSION: &str = "tmp";

/// Distinguishes the temp files of threads in one process publishing the same key.
static NEXT_TORN: AtomicU64 = AtomicU64::new(0);

/// What one open file is, read with `fstat` on the handle that is mapped: its
/// device, inode, length, and its mtime and ctime in nanoseconds since the epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileIdentity {
    pub device: u64,
    pub inode: u64,
    pub length: u64,
    pub mtime_nanos: i128,
    pub ctime_nanos: i128,
}

impl FileIdentity {
    /// `fstat` on `file`, so the identity is of the same file a mapping of it reads.
    #[cfg(unix)]
    pub fn of(file: &File) -> std::io::Result<Self> {
        let metadata = file.metadata()?;
        let nanos = |seconds: i64, sub: i64| i128::from(seconds) * 1_000_000_000 + i128::from(sub);
        Ok(Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            length: metadata.size(),
            mtime_nanos: nanos(metadata.mtime(), metadata.mtime_nsec()),
            ctime_nanos: nanos(metadata.ctime(), metadata.ctime_nsec()),
        })
    }
}

/// The `xxh3` digest naming the store entry for one source checkpoint under one
/// ordered rule list at one alignment. Inputs: the source file's
/// [`FileIdentity`], its header bytes (the length prefix and the JSON), every
/// rule in order (pattern kind, pattern, target), the alignment the store is
/// written at, and [`STORE_LAYOUT_VERSION`]. Every variable-length input is
/// length-prefixed, so no two input sets share a byte stream.
#[must_use]
pub fn store_key(
    header: &[u8],
    identity: &FileIdentity,
    rules: &[WeightPrecisionRule<'_>],
    alignment: usize,
) -> u128 {
    let mut hasher = Xxh3::new();
    hasher.update(&STORE_LAYOUT_VERSION.to_le_bytes());
    hasher.update(&(alignment as u64).to_le_bytes());
    hasher.update(&identity.device.to_le_bytes());
    hasher.update(&identity.inode.to_le_bytes());
    hasher.update(&identity.length.to_le_bytes());
    hasher.update(&identity.mtime_nanos.to_le_bytes());
    hasher.update(&identity.ctime_nanos.to_le_bytes());
    hasher.update(&(header.len() as u64).to_le_bytes());
    hasher.update(header);
    hasher.update(&(rules.len() as u64).to_le_bytes());
    for rule in rules {
        let (kind, pattern) = match rule.pattern {
            NamePattern::Exact(pattern) => (0u8, pattern),
            NamePattern::Prefix(pattern) => (1, pattern),
            NamePattern::Suffix(pattern) => (2, pattern),
        };
        hasher.update(&[kind]);
        hasher.update(&(pattern.len() as u64).to_le_bytes());
        hasher.update(pattern.as_bytes());
        let target = rule.target.name();
        hasher.update(&(target.len() as u64).to_le_bytes());
        hasher.update(target.as_bytes());
    }
    hasher.digest128()
}

/// `bytes`' safetensors header: the eight-byte length prefix and the JSON it
/// declares. `None` when the file is shorter than its own header.
#[must_use]
pub fn header_prefix(bytes: &[u8]) -> Option<&[u8]> {
    let length: [u8; 8] = bytes.get(..8)?.try_into().ok()?;
    let header_len = usize::try_from(u64::from_le_bytes(length)).ok()?;
    bytes.get(..header_len.checked_add(8)?)
}

/// Where the published entry for `key` lives in `dir`.
#[must_use]
pub fn store_path(dir: &Path, key: u128) -> PathBuf {
    dir.join(format!("{key:032x}.{STORE_EXTENSION}"))
}

/// The published entry for `key`, when one exists. A temp file a crashed writer
/// left behind has a different extension and is never returned.
#[must_use]
pub fn lookup(dir: &Path, key: u128) -> Option<PathBuf> {
    Some(store_path(dir, key)).filter(|path| path.is_file())
}

/// Writes `bytes` as the entry for `key`: a temp file in `dir` is written and
/// synced, then renamed over the final name, so a reader sees the whole entry
/// or none of it. The temp name carries the process id, so two processes
/// publishing the same key do not share a temp file; the later rename replaces
/// the earlier with identical bytes.
///
/// # Errors
///
/// [`InteropError::SidecarIo`] when the directory, the temp file or the rename
/// fails; the temp file is removed first.
pub fn publish(dir: &Path, key: u128, bytes: &[u8]) -> Result<PathBuf, InteropError> {
    fs::create_dir_all(dir)?;
    let sequence = NEXT_TORN.fetch_add(1, Ordering::Relaxed);
    let torn = dir.join(format!("{key:032x}.{}.{sequence}.{TORN_EXTENSION}", process::id()));
    let target = store_path(dir, key);
    let written = write_synced(&torn, bytes).and_then(|()| fs::rename(&torn, &target));
    if let Err(error) = written {
        let _ = fs::remove_file(&torn);
        return Err(error.into());
    }
    Ok(target)
}

fn write_synced(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// One `weight_store_hit` event: the key as two `u64` halves, and the bytes of the stored file the
/// load maps instead of the source.
#[cfg_attr(not(feature = "instrument"), allow(unused_variables))]
pub fn emit_weight_store_hit(key: u128, stored_bytes: usize) {
    #[cfg(feature = "instrument")]
    debug!(
        key_high = (key >> 64) as u64,
        key_low = key as u64,
        stored_bytes = stored_bytes as u64,
        "weight_store_hit"
    );
}

/// One `weight_store_publish` event: the key as two `u64` halves, the bytes
/// written, and whether the entry was published.
#[cfg_attr(not(feature = "instrument"), allow(unused_variables))]
pub fn emit_weight_store_publish(key: u128, stored_bytes: usize, ok: bool) {
    #[cfg(feature = "instrument")]
    debug!(
        key_high = (key >> 64) as u64,
        key_low = key as u64,
        stored_bytes = stored_bytes as u64,
        ok = ok,
        "weight_store_publish"
    );
}

/// The alignment a store entry is written at: the backend's no-copy alignment
/// (`omega::backend::no_copy_alignment`), which is the page size where the
/// backend binds mapped weights without a copy and one byte (no padding)
/// where it does not.
#[cfg(feature = "gpu")]
#[must_use]
pub fn device_alignment() -> usize {
    omega::backend::no_copy_alignment()
}

/// Without a GPU backend there is no no-copy path to align for.
#[cfg(not(feature = "gpu"))]
#[must_use]
pub fn device_alignment() -> usize {
    1
}

/// Declares `bytes`, a mapped checkpoint, to the backend so a tensor at an
/// aligned offset inside it binds as an offset view of one no-copy buffer
/// (`omega::backend::register_checkpoint_mapping`). The registration is the
/// calling thread's; `bytes` must stay mapped until [`unregister_mapping`].
#[cfg(feature = "gpu")]
pub fn register_mapping(bytes: &[u8]) {
    omega::backend::register_checkpoint_mapping(bytes);
}

#[cfg(not(feature = "gpu"))]
pub fn register_mapping(_bytes: &[u8]) {}

/// Withdraws a [`register_mapping`]; a no-op for bytes that are not the
/// registered mapping.
#[cfg(feature = "gpu")]
pub fn unregister_mapping(bytes: &[u8]) {
    omega::backend::unregister_checkpoint_mapping(bytes);
}

#[cfg(not(feature = "gpu"))]
pub fn unregister_mapping(_bytes: &[u8]) {}

#[cfg(test)]
mod tests {
    use proxima_gguf::types::GgmlType;
    use proxima_safetensors::{SafetensorsModel, TensorPayload, map_dtype, write_complete_aligned};
    use tempfile::tempdir;

    use super::*;

    const ALIGNMENT: usize = 64;

    fn source_checkpoint() -> Vec<u8> {
        let weights: Vec<u8> = (0..300u32).map(|index| (index % 251) as u8).collect();
        let model = SafetensorsModel {
            tensors: vec![TensorPayload {
                name: "model.embed_tokens.weight".into(),
                dtype: map_dtype("embed", "U8").expect("U8 maps"),
                shape: vec![weights.len() as u64],
                data: &weights,
            }],
            metadata: Default::default(),
        };
        write_complete_aligned(&model, 1).expect("fixture checkpoint writes")
    }

    fn rules() -> Vec<WeightPrecisionRule<'static>> {
        vec![
            WeightPrecisionRule {
                pattern: NamePattern::Suffix("fc1.weight"),
                target: GgmlType::Q8_0,
            },
            WeightPrecisionRule {
                pattern: NamePattern::Exact("proj_out.weight"),
                target: GgmlType::Q4_0,
            },
        ]
    }

    fn identity_of(bytes: &[u8], modified: i128) -> FileIdentity {
        FileIdentity {
            device: 1,
            inode: 10,
            length: bytes.len() as u64,
            mtime_nanos: modified,
            ctime_nanos: 5,
        }
    }

    fn key_of(bytes: &[u8], modified: i128, rules: &[WeightPrecisionRule<'_>]) -> u128 {
        let header = header_prefix(bytes).expect("fixture has a header");
        store_key(header, &identity_of(bytes, modified), rules, ALIGNMENT)
    }

    #[test]
    fn two_files_with_the_same_header_length_and_mtime_never_share_a_key() {
        let bytes = source_checkpoint();
        let header = header_prefix(&bytes).expect("fixture has a header");
        let base = identity_of(&bytes, 7);
        let key = |identity: FileIdentity| store_key(header, &identity, &rules(), ALIGNMENT);

        assert_ne!(key(base), key(FileIdentity { inode: 11, ..base }), "inode");
        assert_ne!(key(base), key(FileIdentity { ctime_nanos: 6, ..base }), "ctime");
        assert_ne!(key(base), key(FileIdentity { device: 2, ..base }), "device");
    }

    #[cfg(unix)]
    #[test]
    fn an_in_place_edit_changes_the_ctime_of_the_open_file() {
        let directory = tempdir().expect("scratch directory");
        let path = directory.path().join("model.safetensors");
        fs::write(&path, source_checkpoint()).expect("writes the checkpoint");
        let file = File::open(&path).expect("opens the checkpoint");
        let before = FileIdentity::of(&file).expect("fstat on the open file");
        let mut edit = fs::OpenOptions::new().write(true).open(&path).expect("reopens for write");

        let mut after = before;
        while after.ctime_nanos == before.ctime_nanos {
            edit.write_all(&[1]).expect("edits in place");
            edit.sync_all().expect("syncs the edit");
            after = FileIdentity::of(&file).expect("fstat on the open file");
        }

        assert_eq!(after.inode, before.inode, "the same file");
        assert_ne!(after.ctime_nanos, before.ctime_nanos);
    }

    #[test]
    fn store_key_is_stable_for_the_same_inputs() {
        let bytes = source_checkpoint();

        assert_eq!(key_of(&bytes, 7, &rules()), key_of(&bytes, 7, &rules()));
    }

    #[test]
    fn store_key_changes_with_the_rule_order_the_target_and_the_pattern() {
        let bytes = source_checkpoint();
        let base = key_of(&bytes, 7, &rules());
        let mut reordered = rules();
        reordered.reverse();
        let mut retargeted = rules();
        retargeted[0].target = GgmlType::Q4_0;
        let mut repatterned = rules();
        repatterned[0].pattern = NamePattern::Prefix("fc1.weight");

        assert_ne!(base, key_of(&bytes, 7, &reordered), "rule order");
        assert_ne!(base, key_of(&bytes, 7, &retargeted), "rule target");
        assert_ne!(base, key_of(&bytes, 7, &repatterned), "pattern kind");
        assert_ne!(base, key_of(&bytes, 7, &[]), "no rules");
    }

    #[test]
    fn store_key_changes_with_a_header_byte_the_length_the_mtime_and_the_alignment() {
        let bytes = source_checkpoint();
        let base = key_of(&bytes, 7, &rules());
        let mut edited_header = bytes.clone();
        edited_header[8 + 2] ^= 1;
        let header = header_prefix(&bytes).expect("fixture has a header");

        assert_ne!(base, key_of(&edited_header, 7, &rules()), "header byte");
        let longer = FileIdentity {
            length: bytes.len() as u64 + 1,
            ..identity_of(&bytes, 7)
        };
        assert_ne!(base, store_key(header, &longer, &rules(), ALIGNMENT), "file length");
        assert_ne!(base, key_of(&bytes, 8, &rules()), "modification time");
        assert_ne!(
            base,
            store_key(header, &identity_of(&bytes, 7), &rules(), ALIGNMENT * 2),
            "alignment"
        );
    }

    #[test]
    fn a_published_entry_is_found_and_holds_the_published_bytes() {
        let directory = tempdir().expect("scratch store directory");
        let store = directory.path().join("store");
        let bytes = source_checkpoint();
        let key = key_of(&bytes, 7, &rules());

        assert_eq!(lookup(&store, key), None, "nothing is published yet");
        let published = publish(&store, key, &bytes).expect("publishes");

        assert_eq!(lookup(&store, key), Some(published.clone()));
        assert_eq!(fs::read(published).expect("reads the entry"), bytes);
    }

    #[test]
    fn a_torn_temp_file_is_never_returned_as_an_entry() {
        let directory = tempdir().expect("scratch store directory");
        let bytes = source_checkpoint();
        let key = key_of(&bytes, 7, &rules());
        let torn = directory
            .path()
            .join(format!("{key:032x}.1234.{TORN_EXTENSION}"));
        fs::write(&torn, &bytes[..bytes.len() - 5]).expect("writes a truncated entry");

        assert_eq!(lookup(directory.path(), key), None);
    }

    #[test]
    fn publish_leaves_no_temp_file_behind() {
        let directory = tempdir().expect("scratch store directory");
        let bytes = source_checkpoint();
        let key = key_of(&bytes, 7, &rules());

        publish(directory.path(), key, &bytes).expect("publishes");

        let leftovers = fs::read_dir(directory.path())
            .expect("lists the store")
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == TORN_EXTENSION)
            })
            .count();
        assert_eq!(leftovers, 0);
    }

    #[test]
    fn header_prefix_is_none_when_the_file_is_shorter_than_its_header() {
        let bytes = source_checkpoint();

        assert!(header_prefix(&bytes[..12]).is_none());
        assert!(header_prefix(&bytes[..4]).is_none());
    }
}
