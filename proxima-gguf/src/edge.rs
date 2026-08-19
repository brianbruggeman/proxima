//! The one `std`-gated surface in this crate: reading a whole file into
//! memory and handing its bytes to the sans-IO parser. Deliberately not an
//! `mmap` — see `lib.rs`'s module doc for why. A caller who wants
//! zero-copy weight loading mmaps the file themselves (any mmap crate, or
//! `rustix::mm` directly, as `proxima-storage/src/dax/region.rs` does for
//! its own domain) and feeds the resulting `&[u8]` straight into
//! [`crate::parser::GgufParser`] or [`crate::pipe::parse_complete`] — this
//! module exists only for the common "I just want the bytes" case.

use std::path::Path;

use thiserror::Error;

use crate::error::GgufError;
use crate::pipe::{ParsedGguf, parse_complete};

/// Failures specific to the file-reading edge, on top of [`GgufError`].
#[derive(Debug, Error)]
pub enum EdgeError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    Parse(#[from] GgufError),
}

/// Reads `path` fully into memory and parses it. Returns the parsed
/// metadata alongside the owned byte buffer so the caller can slice tensor
/// data out of it via [`ParsedGguf::tensor_data_range`] without a second
/// read.
///
/// # Errors
///
/// [`EdgeError::Io`] if the file can't be read; [`EdgeError::Parse`] for
/// any malformed-GGUF condition the parser catches.
pub fn read_file(path: &Path) -> Result<(ParsedGguf, Vec<u8>), EdgeError> {
    let bytes = std::fs::read(path).map_err(|source| EdgeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let parsed = parse_complete(&bytes)?;
    Ok((parsed, bytes))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::io::Write;

    use super::*;

    #[test]
    fn read_file_surfaces_io_error_for_missing_path() {
        let missing = Path::new("/nonexistent/definitely-not-here.gguf");
        let outcome = read_file(missing);
        assert!(matches!(outcome, Err(EdgeError::Io { .. })));
    }

    #[test]
    fn read_file_parses_a_written_synthetic_file() {
        let dir = tempfile::tempdir().expect("tempdir for synthetic gguf fixture");
        let path = dir.path().join("synthetic.gguf");
        let bytes = crate::tests::synthetic_gguf();
        let mut file = std::fs::File::create(&path).expect("create synthetic gguf file");
        file.write_all(&bytes).expect("write synthetic gguf bytes");
        drop(file);

        let (parsed, owned_bytes) = read_file(&path).expect("parse synthetic gguf file");
        assert_eq!(owned_bytes, bytes);
        assert_eq!(parsed.version, 3);
    }

    /// Looks for a real `.gguf` under `~/repos/others/llama.cpp/models` (the
    /// vocab-only fixtures llama.cpp ships in-tree) and, if found, parses it
    /// and prints architecture, tensor count, and the first few tensor
    /// names. `#[ignore]`d: it depends on a sibling checkout that may not
    /// exist on every host, and the suite must not fail when it's absent.
    #[test]
    #[ignore = "depends on a real .gguf checkout outside this repo"]
    fn parses_a_real_gguf_file_if_one_is_present() {
        let candidate = Path::new(
            "/Users/brianbruggeman/repos/others/llama.cpp/models/ggml-vocab-llama-bpe.gguf",
        );
        if !candidate.exists() {
            eprintln!("no real .gguf found at {candidate:?}, skipping");
            return;
        }
        let (parsed, _bytes) = read_file(candidate).expect("parse real gguf file");
        let architecture = parsed
            .metadata_value("general.architecture")
            .and_then(crate::value::MetadataValue::as_str)
            .unwrap_or("<unknown>");
        println!("architecture: {architecture}");
        println!("tensor_count: {}", parsed.tensors.len());
        for tensor in parsed.tensors.iter().take(5) {
            println!("tensor: {}", tensor.name);
        }
    }
}
