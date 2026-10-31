use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::oci::digest::Digest;
use crate::oci::error::OciError;

fn blob_path(root: &Path, digest: &Digest) -> PathBuf {
    root.join("blobs").join("sha256").join(digest.hex())
}

fn failure(path: &Path, source: io::Error) -> OciError {
    OciError::Cache { path: path.to_path_buf(), source }
}

pub fn read(root: &Path, digest: &Digest) -> Result<Option<Vec<u8>>, OciError> {
    let path = blob_path(root, digest);
    match fs::read(&path) {
        Ok(bytes) => {
            digest.verify(&format!("cached blob {}", path.display()), &bytes)?;
            Ok(Some(bytes))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(failure(&path, error)),
    }
}

pub fn write(root: &Path, digest: &Digest, bytes: &[u8]) -> Result<(), OciError> {
    let path = blob_path(root, digest);
    let directory = path.parent().ok_or_else(|| failure(&path, io::Error::other("blob path has no parent")))?;
    fs::create_dir_all(directory).map_err(|error| failure(directory, error))?;
    let staged = directory.join(format!("{}.partial-{}", digest.hex(), std::process::id()));
    fs::write(&staged, bytes).map_err(|error| failure(&staged, error))?;
    fs::rename(&staged, &path).map_err(|error| failure(&path, error))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn a_written_blob_reads_back_and_an_absent_one_is_none() {
        let root = tempfile::tempdir().expect("tempdir");
        let digest = Digest::of(b"layer bytes");

        assert!(read(root.path(), &digest).expect("absent is not an error").is_none());
        write(root.path(), &digest, b"layer bytes").expect("write");

        assert_eq!(read(root.path(), &digest).expect("read").expect("present"), b"layer bytes");
    }

    #[test]
    fn a_blob_altered_on_disk_is_rejected_with_both_digests() {
        let root = tempfile::tempdir().expect("tempdir");
        let digest = Digest::of(b"layer bytes");
        write(root.path(), &digest, b"layer bytes").expect("write");
        fs::write(blob_path(root.path(), &digest), b"tampered").expect("tamper");

        let error = read(root.path(), &digest).expect_err("tampered blob");

        let message = error.to_string();
        assert!(message.contains(&format!("expected {digest}")) && message.contains(&format!("actual {}", Digest::of(b"tampered"))), "{message}");
    }
}
