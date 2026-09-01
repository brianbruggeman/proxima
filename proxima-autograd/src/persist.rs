//! Bridges [`train::State`](crate::train::State) to a safetensors checkpoint
//! on disk through [`proxima_safetensors::write_complete`] /
//! [`proxima_safetensors::SafetensorsParser`] -- the two existing sans-IO
//! primitives this whole module composes.
//!
//! `State` is `Vec<(String, Vec<f32>)>`: a name and its flat buffer, no
//! shape (see `train.rs`'s own doc for why it stays that shape). Safetensors
//! needs a shape per tensor, so [`save_state`]/[`load_state`] take the
//! trained `program: &[Op]` as a second read of shape -- the same
//! [`Op::Input`] declarations [`crate::train::train_step`] already binds
//! `named` against -- rather than widening `State` into a second type that
//! carries shape everywhere it is threaded through a training loop.
//!
//! Call site, both ways, is the check this module was held to before it was
//! written (guiding principle 1): without this module, saving a trained
//! network is "hand-build a `SafetensorsModel` from `state` and `program`,
//! call `write_complete`, `std::fs::write`" repeated at every call site
//! that trains something; with it, `save_state(&program, &state, path)?`.
//! [`load_state`] is the same shape in reverse, plus the two checks a raw
//! `write_complete`/parse round-trip does not give you for free: every
//! parameter the program declares is present, and its shape matches.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::path::Path;

use proxima_safetensors::{HEADER_LEN_BYTES, SafetensorsError, SafetensorsModel, SafetensorsParser, TensorPayload};
use proxima_tensor::DType;
use proxima_tensor::op::{Extent, Op};

use crate::train::State;

/// Every fault [`save_state`]/[`load_state`] can raise: this crate's own
/// program/state bookkeeping, [`proxima_safetensors::SafetensorsError`] from
/// the wire codec, and `std::io::Error` from the file itself.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PersistError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("safetensors: {0}")]
    Safetensors(#[from] SafetensorsError),

    #[error("parameter {name} has no Op::Input declaration in the program handed to save_state")]
    UndeclaredParameter { name: String },

    #[error(
        "parameter {name} has a symbolic axis (symbol {symbol}) in its declared shape -- \
         save_state only saves parameters whose shape is fully static"
    )]
    SymbolicShape { name: String, symbol: u16 },

    #[error("checkpoint at {path} is missing parameter {name}, which the program declares")]
    MissingParameter { path: String, name: String },

    #[error("parameter {name} shape mismatch: program declares {expected:?}, checkpoint has {found:?}")]
    ShapeMismatch { name: String, expected: Vec<u64>, found: Vec<u64> },

    #[error("checkpoint at {path} is shorter than its own declared header length")]
    TruncatedTensorData { path: String },
}

/// Looks up `name`'s declared shape among `program`'s [`Op::Input`] leaves,
/// resolving every [`Extent::Static`] axis to a `u64` -- the same shape a
/// [`Op::Input`] leaf declares for [`proxima_tensor::cpu::evaluate_named`]
/// to bind against, read here rather than re-derived through
/// [`proxima_tensor::shape::infer`] (that pass resolves the whole program's
/// symbolic axes; a trained parameter's own leaf shape is already fully
/// static, so reading its declaration directly is the smaller true answer).
fn declared_shape(program: &[Op], name: &str) -> Result<Vec<u64>, PersistError> {
    let shape = program
        .iter()
        .find_map(|op| match op {
            Op::Input { name: Some(candidate), shape, .. } if candidate == name => Some(shape),
            _ => None,
        })
        .ok_or_else(|| PersistError::UndeclaredParameter { name: name.to_string() })?;

    shape
        .iter()
        .map(|extent| match extent {
            Extent::Static(size) => Ok(u64::from(*size)),
            Extent::Symbolic(symbol) => Err(PersistError::SymbolicShape { name: name.to_string(), symbol: *symbol }),
        })
        .collect()
}

/// Writes `state` to `path` as one safetensors file: every `(name, values)`
/// pair becomes one tensor, `f32`, little-endian, shaped by `name`'s
/// [`Op::Input`] declaration in `program`.
///
/// # Errors
///
/// [`PersistError::UndeclaredParameter`]/[`PersistError::SymbolicShape`] if
/// `program` has no fully-static [`Op::Input`] declaration for a name in
/// `state`; [`PersistError::Safetensors`] for a
/// [`proxima_safetensors::write_complete`] fault (duplicate/reserved name,
/// length mismatch); [`PersistError::Io`] if `path` cannot be written.
pub fn save_state(program: &[Op], state: &State, path: &Path) -> Result<(), PersistError> {
    let mut byte_buffers: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::with_capacity(state.len());
    for (name, values) in state {
        let shape = declared_shape(program, name)?;
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        byte_buffers.push((name.clone(), shape, bytes));
    }

    let tensors: Vec<TensorPayload<'_>> = byte_buffers
        .iter()
        .map(|(name, shape, bytes)| TensorPayload {
            name: name.clone(),
            dtype: DType::Float32,
            shape: shape.clone(),
            data: bytes.as_slice(),
        })
        .collect();
    let model = SafetensorsModel { tensors, metadata: alloc::collections::BTreeMap::new() };
    let wire = proxima_safetensors::write_complete(&model)?;
    std::fs::write(path, wire)?;
    Ok(())
}

/// Reads `path` back into a [`State`], validating every name `program`
/// declares is present with a matching shape before handing back a single
/// `f32` buffer per tensor -- the checks a bare
/// [`proxima_safetensors::SafetensorsParser`] round-trip leaves to the
/// caller.
///
/// Only the names `program` declares (via [`Op::Input`]) are read back; a
/// checkpoint carrying extra tensors the program does not use is not an
/// error, mirroring [`proxima_safetensors::Manifest`] itself never
/// rejecting an unrecognized entry.
///
/// # Errors
///
/// [`PersistError::Io`] if `path` cannot be read; [`PersistError::Safetensors`]
/// if the bytes are not a well-formed safetensors file, or if
/// [`proxima_safetensors::Manifest::format_version`] rejects the
/// checkpoint's `__metadata__` format-version stamp (a checkpoint with no
/// stamp at all -- every one this crate wrote before the stamp existed --
/// is accepted, not rejected);
/// [`PersistError::UndeclaredParameter`]/[`PersistError::SymbolicShape`] for
/// the same program-shape faults [`save_state`] raises;
/// [`PersistError::MissingParameter`] if `program` declares a name the
/// checkpoint does not carry; [`PersistError::ShapeMismatch`] if the
/// checkpoint's shape for a name disagrees with `program`'s declaration.
pub fn load_state(program: &[Op], path: &Path) -> Result<State, PersistError> {
    let path_display = path.display().to_string();
    let bytes = std::fs::read(path)?;
    let manifest = SafetensorsParser::new().push(&bytes)?.finish()?;
    manifest.format_version()?;

    let header_len_start = HEADER_LEN_BYTES;
    let header_len = bytes
        .get(..header_len_start)
        .and_then(|slice| <[u8; 8]>::try_from(slice).ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| PersistError::TruncatedTensorData { path: path_display.clone() })?;
    let data_start = header_len_start + header_len as usize;
    let data_region = bytes
        .get(data_start..)
        .ok_or_else(|| PersistError::TruncatedTensorData { path: path_display.clone() })?;

    let mut declared_names: Vec<&str> = Vec::new();
    for op in program {
        if let Op::Input { name: Some(name), .. } = op
            && !declared_names.contains(&name.as_str())
        {
            declared_names.push(name.as_str());
        }
    }

    let mut state = Vec::with_capacity(declared_names.len());
    for name in declared_names {
        let expected_shape = declared_shape(program, name)?;
        let entry = manifest.tensor(name).ok_or_else(|| PersistError::MissingParameter {
            path: path_display.clone(),
            name: name.to_string(),
        })?;
        if entry.shape != expected_shape {
            return Err(PersistError::ShapeMismatch {
                name: name.to_string(),
                expected: expected_shape,
                found: entry.shape.clone(),
            });
        }

        let (start, end) = entry.data_offsets;
        let raw = data_region
            .get(start as usize..end as usize)
            .ok_or_else(|| PersistError::TruncatedTensorData { path: path_display.clone() })?;
        let values: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect();
        state.push((name.to_string(), values));
    }

    Ok(state)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use proxima_tensor::dtype::DType as TensorDType;
    use proxima_tensor::op::{self, Extent, NodeId};

    use super::*;

    fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
        op::append(program, Op::Input { dtype: TensorDType::Float32, shape, name: Some(name.into()) })
    }

    fn toy_program() -> Vec<Op> {
        let mut program = Vec::new();
        leaf(&mut program, "w", vec![Extent::Static(2), Extent::Static(3)]);
        leaf(&mut program, "b", vec![Extent::Static(3)]);
        program
    }

    fn toy_state() -> State {
        vec![(String::from("w"), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), (String::from("b"), vec![0.1, 0.2, 0.3])]
    }

    /// Hand-builds a safetensors wire buffer the same way
    /// `proxima_safetensors::tests::build_buffer` does, bypassing
    /// [`proxima_safetensors::write_complete`]'s always-on format-version
    /// stamp -- the only way to reproduce a checkpoint byte-for-byte as
    /// this crate's `save_state` would have written it before that stamp
    /// existed, or to plant an arbitrary tampered stamp for the
    /// unsupported-version test below.
    fn hand_written_checkpoint_bytes(entries: &[(&str, &[u64], &[f32])], metadata: &[(&str, &str)]) -> Vec<u8> {
        let mut header = String::from("{");
        if !metadata.is_empty() {
            header.push_str("\"__metadata__\":{");
            for (index, (key, value)) in metadata.iter().enumerate() {
                if index > 0 {
                    header.push(',');
                }
                header.push_str(&alloc::format!("{key:?}:{value:?}"));
            }
            header.push_str("},");
        }

        let mut data = Vec::new();
        for (index, (name, shape, values)) in entries.iter().enumerate() {
            if index > 0 {
                header.push(',');
            }
            let start = data.len() as u64;
            for value in values.iter() {
                data.extend_from_slice(&value.to_le_bytes());
            }
            let end = data.len() as u64;
            let shape_json = shape.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
            header.push_str(&alloc::format!(
                "{name:?}:{{\"dtype\":\"F32\",\"shape\":[{shape_json}],\"data_offsets\":[{start},{end}]}}"
            ));
        }
        header.push('}');

        let header_bytes = header.into_bytes();
        let mut wire = Vec::new();
        wire.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        wire.extend_from_slice(&header_bytes);
        wire.extend_from_slice(&data);
        wire
    }

    #[test]
    fn save_then_load_round_trips_every_named_buffer_exactly() {
        let program = toy_program();
        let state = toy_state();
        let file = tempfile::NamedTempFile::new().expect("create temp checkpoint file");

        save_state(&program, &state, file.path()).expect("save_state writes the checkpoint");
        let loaded = load_state(&program, file.path()).expect("load_state reads it back");

        assert_eq!(loaded.len(), state.len());
        for (name, values) in &state {
            let found = loaded.iter().find(|(candidate, _)| candidate == name).map(|(_, values)| values).expect("name present after round trip");
            assert_eq!(found, values, "buffer for {name} must round-trip exactly");
        }
    }

    #[test]
    fn save_then_parse_with_the_safetensors_crate_itself_round_trips_names_shapes_and_dtypes() {
        let program = toy_program();
        let state = toy_state();
        let file = tempfile::NamedTempFile::new().expect("create temp checkpoint file");

        save_state(&program, &state, file.path()).expect("save_state writes the checkpoint");
        let bytes = std::fs::read(file.path()).expect("read written checkpoint");
        let manifest = SafetensorsParser::new().push(&bytes).expect("parser accepts the bytes").finish().expect("manifest parses");

        let w_entry = manifest.tensor("w").expect("w present in the manifest");
        assert_eq!(w_entry.dtype, proxima_tensor::DType::Float32);
        assert_eq!(w_entry.shape, vec![2, 3]);

        let b_entry = manifest.tensor("b").expect("b present in the manifest");
        assert_eq!(b_entry.dtype, proxima_tensor::DType::Float32);
        assert_eq!(b_entry.shape, vec![3]);
    }

    #[test]
    fn save_state_rejects_a_state_name_with_no_matching_op_input_declaration() {
        let program = toy_program();
        let mut state = toy_state();
        state.push((String::from("ghost"), vec![9.0]));
        let file = tempfile::NamedTempFile::new().expect("create temp checkpoint file");

        let outcome = save_state(&program, &state, file.path());
        assert!(
            matches!(outcome, Err(PersistError::UndeclaredParameter { ref name }) if name == "ghost"),
            "expected an UndeclaredParameter for ghost, got {outcome:?}"
        );
    }

    #[test]
    fn load_state_rejects_a_checkpoint_with_a_mismatched_shape() {
        let save_program = toy_program();
        let state = toy_state();
        let file = tempfile::NamedTempFile::new().expect("create temp checkpoint file");
        save_state(&save_program, &state, file.path()).expect("save_state writes the checkpoint");

        let mut load_program = Vec::new();
        leaf(&mut load_program, "w", vec![Extent::Static(3), Extent::Static(2)]);
        leaf(&mut load_program, "b", vec![Extent::Static(3)]);

        let outcome = load_state(&load_program, file.path());
        assert!(
            matches!(outcome, Err(PersistError::ShapeMismatch { ref name, .. }) if name == "w"),
            "expected a ShapeMismatch for w, got {outcome:?}"
        );
    }

    #[test]
    fn load_state_rejects_a_checkpoint_missing_a_parameter_the_program_declares() {
        let mut save_program = Vec::new();
        leaf(&mut save_program, "w", vec![Extent::Static(2), Extent::Static(3)]);
        let state = vec![(String::from("w"), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])];
        let file = tempfile::NamedTempFile::new().expect("create temp checkpoint file");
        save_state(&save_program, &state, file.path()).expect("save_state writes the checkpoint");

        let load_program = toy_program(); // declares both "w" and "b"
        let outcome = load_state(&load_program, file.path());
        assert!(
            matches!(outcome, Err(PersistError::MissingParameter { ref name, .. }) if name == "b"),
            "expected a MissingParameter for b, got {outcome:?}"
        );
    }

    /// Every checkpoint `save_state` ever wrote before
    /// [`proxima_safetensors::write_complete`] started stamping a
    /// format-version key must keep loading -- this fixture reproduces
    /// that exact shape (no `__metadata__` entry at all) rather than
    /// relying on `save_state` itself, which now always stamps.
    #[test]
    fn load_state_accepts_a_pre_change_checkpoint_carrying_no_format_version_stamp() {
        let program = toy_program();
        let state = toy_state();
        let bytes = hand_written_checkpoint_bytes(
            &[("w", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), ("b", &[3], &[0.1, 0.2, 0.3])],
            &[],
        );
        let file = tempfile::NamedTempFile::new().expect("create temp checkpoint file");
        std::fs::write(file.path(), &bytes).expect("write pre-change-shaped checkpoint bytes");

        let loaded = load_state(&program, file.path()).expect("pre-change checkpoint with no version stamp still loads");
        assert_eq!(loaded, state, "pre-change checkpoint must load byte-for-byte identical values");
    }

    #[test]
    fn load_state_rejects_a_checkpoint_declaring_an_unsupported_major_format_version() {
        let program = toy_program();
        let unknown_major = proxima_safetensors::FORMAT_VERSION_MAJOR + 1;
        let unsupported_stamp = alloc::format!("{unknown_major}.0");
        let bytes = hand_written_checkpoint_bytes(
            &[("w", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), ("b", &[3], &[0.1, 0.2, 0.3])],
            &[(proxima_safetensors::FORMAT_VERSION_KEY, unsupported_stamp.as_str())],
        );
        let file = tempfile::NamedTempFile::new().expect("create temp checkpoint file");
        std::fs::write(file.path(), &bytes).expect("write checkpoint with an unsupported version stamp");

        let outcome = load_state(&program, file.path());
        assert!(
            matches!(
                outcome,
                Err(PersistError::Safetensors(SafetensorsError::UnsupportedFormatVersion { ref found, .. }))
                    if found == &unsupported_stamp
            ),
            "expected an UnsupportedFormatVersion naming {unsupported_stamp:?}, got {outcome:?}"
        );
    }
}
