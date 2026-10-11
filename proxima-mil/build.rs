use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;

use proxima_build::sizing::{SizingError, SizingSource, require_nonzero};

#[derive(Debug, thiserror::Error)]
enum BuildError {
    #[error("sizing: {0}")]
    Sizing(#[from] SizingError),
    #[error("{0} is not set by cargo")]
    EnvUnset(&'static str),
    #[error("write {path}: {source}", path = .path.display())]
    Write { path: PathBuf, source: io::Error },
}

fn main() -> Result<(), BuildError> {
    let manifest_dir =
        env::var("CARGO_MANIFEST_DIR").map_err(|_| BuildError::EnvUnset("CARGO_MANIFEST_DIR"))?;
    let out_dir = env::var("OUT_DIR").map_err(|_| BuildError::EnvUnset("OUT_DIR"))?;
    let source = SizingSource::load(&manifest_dir, "proxima-mil.toml", "PROXIMA_MIL")?;
    let module = render_module(&source)?;
    let path = PathBuf::from(out_dir).join("proxima_mil_sized.rs");
    fs::write(&path, module).map_err(|source| BuildError::Write { path, source })
}

fn render_module(source: &SizingSource) -> Result<String, BuildError> {
    let graph_ceiling_mb = require_nonzero(
        "ane.graph_ceiling_mb",
        source.resolve_int("ane", "graph_ceiling_mb")?,
    );
    let pad_multiple = require_nonzero(
        "shape.pad_multiple",
        source.resolve_int("shape", "pad_multiple")?,
    );
    let max_sequence = require_nonzero(
        "shape.max_sequence",
        source.resolve_int("shape", "max_sequence")?,
    );
    let specification_version = require_nonzero(
        "package.specification_version",
        source.resolve_int("package", "specification_version")?,
    );
    let opset = source.resolve_str("package", "opset")?;
    let default_compute_units = source.resolve_str("units", "default_compute_units")?;
    Ok(format!(
        "pub const GRAPH_CEILING_MB: usize = {graph_ceiling_mb};\n\
         pub const PAD_MULTIPLE: usize = {pad_multiple};\n\
         pub const MAX_SEQUENCE: usize = {max_sequence};\n\
         pub const SPECIFICATION_VERSION: usize = {specification_version};\n\
         pub const OPSET: &str = {opset:?};\n\
         pub const DEFAULT_COMPUTE_UNITS: &str = {default_compute_units:?};\n"
    ))
}
