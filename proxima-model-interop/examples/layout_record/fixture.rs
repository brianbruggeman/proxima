use std::fs;
use std::path::Path;

use proxima_primitives::Codec;
use sha2::{Digest, Sha256};

use crate::error::RecordError;
use crate::incumbent::ELEMENTS;

pub enum Origin {
    Checkpoint(String),
    Encoder(String),
}

pub struct Recorded {
    pub codec: Codec,
    pub origin: Origin,
    pub blocks: Vec<u8>,
    pub decoded: Vec<f32>,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn decoded_bytes(decoded: &[f32]) -> Vec<u8> {
    decoded
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn label(origin: &Origin) -> (&'static str, &str) {
    match origin {
        Origin::Checkpoint(detail) => ("checkpoint", detail),
        Origin::Encoder(detail) => ("encoder", detail),
    }
}

fn write_file(directory: &Path, name: &str, bytes: &[u8]) -> Result<(), RecordError> {
    let path = directory.join(name);
    fs::write(&path, bytes).map_err(RecordError::io(&path))
}

pub fn write(out: &Path, recorded: &Recorded) -> Result<String, RecordError> {
    let directory = out.join(format!("{:?}", recorded.codec));
    fs::create_dir_all(&directory).map_err(RecordError::io(&directory))?;
    let decoded = decoded_bytes(&recorded.decoded);
    let decoded_sha = sha256_hex(&decoded);
    let (source, detail) = label(&recorded.origin);
    write_file(&directory, "blocks.bin", &recorded.blocks)?;
    write_file(&directory, "decoded_f32.bin", &decoded)?;
    let manifest = format!(
        "codec = \"{:?}\"\nelements = {ELEMENTS}\nsource = \"{source}\"\norigin = \"{detail}\"\nblocks_sha256 = \"{}\"\ndecoded_f32_sha256 = \"{decoded_sha}\"\n",
        recorded.codec,
        sha256_hex(&recorded.blocks),
    );
    write_file(&directory, "manifest.toml", manifest.as_bytes())?;
    Ok(format!(
        "fixture codec={:?} elements={ELEMENTS} source={source} sha256={decoded_sha}",
        recorded.codec
    ))
}

pub fn write_unavailable(out: &Path, codec: Codec) -> Result<String, RecordError> {
    let directory = out.join(format!("{codec:?}"));
    fs::create_dir_all(&directory).map_err(RecordError::io(&directory))?;
    let manifest = format!(
        "codec = \"{codec:?}\"\nstatus = \"unavailable\"\nreason = \"no_incumbent_decoder\"\n"
    );
    write_file(&directory, "manifest.toml", manifest.as_bytes())?;
    Ok(format!(
        "fixture codec={codec:?} unavailable reason=no_incumbent_decoder"
    ))
}
