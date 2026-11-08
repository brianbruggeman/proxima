//! The files of one MLX model directory for a GGUF source: weight shards and
//! their index, `config.json`, `tokenizer.json` and `tokenizer_config.json`.
//!
//! [`export_mlx_directory`] plans the weights with
//! [`super::mlx_weight_plan::WeightPlan`], fixes the shard cut and every
//! safetensors header with [`super::MlxLayout`] before a weight is read, then
//! converts a window of row blocks at a time and writes each shard as it
//! goes, so the process holds the blocks in flight and the `scales` and
//! `biases` of the tensor being written, never the model. The small files come
//! from [`super::hf_config_bytes_with_overrides`] and [`super::export_tokenizer`].

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::fs::{File, create_dir_all, write};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_gguf::pipe::ParsedGguf;
use proxima_safetensors::TensorHeader;
use proxima_numeric::DType;

use super::mlx::{MLX_INDEX_FILE_NAME, MLX_MAX_SHARD_BYTES, MlxLayout, MlxShardPlan};
use super::mlx_affine::AffineParams;
use super::mlx_config::{family_export, hf_config_bytes_with_overrides};
use super::mlx_tokenizer::export_tokenizer;
use super::mlx_weight_plan::{WeightChunk, WeightPlan, available_workers};
use crate::error::InteropError;
use crate::profiles::binding_profile;

const CONFIG_FILE_NAME: &str = "config.json";
const TOKENIZER_FILE_NAME: &str = "tokenizer.json";
const TOKENIZER_CONFIG_FILE_NAME: &str = "tokenizer_config.json";
const WRITE_BUFFER_BYTES: usize = 1 << 20;

fn config_marker(params: AffineParams) -> Result<(u8, u32), InteropError> {
    let group_size = params.group_size();
    u32::try_from(group_size)
        .map(|group| (params.bits(), group))
        .map_err(|_| InteropError::AffineGroupSizeUnsupported { group_size })
}

fn io_error(path: &Path) -> impl FnOnce(std::io::Error) -> InteropError {
    let path = path.to_path_buf();
    move |source| InteropError::MlxDirectoryIo { path, source }
}

/// Maps the GGUF at `gguf_path` and writes its MLX directory under `out_dir`
/// (created if absent, existing files overwritten), quantized with
/// `quantization` (plain float tensors as `dtype` when `None`); returns the
/// paths written.
///
/// # Errors
///
/// [`InteropError::MlxDirectoryIo`] naming the path that could not be opened,
/// mapped, created or written; [`InteropError::UnknownFamilyConfig`] for a
/// family without an MLX layout, before anything is created; otherwise
/// whatever the composed planner, converter, config and tokenizer writers fail
/// with.
pub fn export_mlx_directory(
    gguf_path: &Path,
    out_dir: &Path,
    quantization: Option<AffineParams>,
    dtype: DType,
) -> Result<Vec<PathBuf>, InteropError> {
    let file = File::open(gguf_path).map_err(io_error(gguf_path))?;
    // the checkpoint is a read-only blob that is not rewritten during an export
    let mapping = unsafe { Mmap::map(&file) }.map_err(io_error(gguf_path))?;
    let parsed = parse_complete(&mapping)?;
    let (names_layout, hf_prefix, conventions, hparams) = family_export(&parsed)?;
    let binding = binding_profile(&hparams.family)?;
    let plan = WeightPlan::new(
        &parsed,
        &mapping,
        dtype,
        &binding,
        names_layout,
        hf_prefix,
        quantization.map(|params| (params, conventions)),
    )?;
    let headers = plan.headers()?;
    let overrides = quantization.map_or_else(Vec::new, |params| {
        conventions.overrides(headers.iter().map(|header| header.name.as_str()), params)
    });
    let marker = quantization.map(config_marker).transpose()?;
    let [tokenizer, tokenizer_config] = tokenizer_files(&parsed)?;
    let small_files = [
        (
            CONFIG_FILE_NAME.to_string(),
            hf_config_bytes_with_overrides(&hparams, &parsed, marker, &overrides)?,
        ),
        tokenizer,
        tokenizer_config,
    ];
    let layout = MlxLayout::plan(headers, MLX_MAX_SHARD_BYTES)?;
    proxima_telemetry::debug!(
        shard_count = layout.shards.len() as u64,
        tensor_count = layout
            .shards
            .iter()
            .map(|shard| shard.tensors.len() as u64)
            .sum::<u64>(),
        header_bytes = layout
            .shards
            .iter()
            .map(|shard| shard.prefix.len() as u64)
            .sum::<u64>(),
        "mlx weight layout planned before any weight is converted"
    );
    create_dir_all(out_dir).map_err(io_error(out_dir))?;
    let mut paths = write_weights(out_dir, &plan, &layout)?;
    if let Some(index) = &layout.index {
        let path = out_dir.join(MLX_INDEX_FILE_NAME);
        write(&path, index).map_err(io_error(&path))?;
        paths.push(path);
    }
    for (name, bytes) in small_files {
        let path = out_dir.join(name);
        write(&path, bytes).map_err(io_error(&path))?;
        paths.push(path);
    }
    Ok(paths)
}

fn tokenizer_files(parsed: &ParsedGguf) -> Result<[(String, Vec<u8>); 2], InteropError> {
    let tokenizer = export_tokenizer(parsed)?;
    Ok([
        (TOKENIZER_FILE_NAME.to_string(), tokenizer.tokenizer_json),
        (
            TOKENIZER_CONFIG_FILE_NAME.to_string(),
            tokenizer.tokenizer_config_json,
        ),
    ])
}

fn write_weights(
    out_dir: &Path,
    plan: &WeightPlan,
    layout: &MlxLayout,
) -> Result<Vec<PathBuf>, InteropError> {
    let workers = available_workers();
    let mut sink = ShardSink::new(out_dir, &layout.shards);
    for window in plan.blocks().chunks(workers) {
        let chunks = plan.convert(window, workers)?;
        proxima_telemetry::debug!(
            window_blocks = window.len() as u64,
            chunk_bytes = chunks.iter().map(chunk_bytes).sum::<u64>(),
            held_bytes = sink.held_bytes(),
            written_bytes = sink.written_bytes,
            "mlx window converted and about to be written"
        );
        chunks.into_iter().try_for_each(|chunk| sink.push(chunk))?;
    }
    sink.finish()
}

fn chunk_bytes(chunk: &WeightChunk) -> u64 {
    chunk.parts.iter().map(|part| part.len() as u64).sum()
}

struct OpenShard<'plan> {
    path: PathBuf,
    writer: BufWriter<File>,
    pending: &'plan [TensorHeader],
}

struct ActiveTensor<'plan> {
    header: &'plan TensorHeader,
    written: u64,
}

struct ShardSink<'plan> {
    out_dir: &'plan Path,
    shards: &'plan [MlxShardPlan],
    next_shard: usize,
    open: Option<OpenShard<'plan>>,
    active: Option<ActiveTensor<'plan>>,
    held: Vec<Vec<u8>>,
    written_bytes: u64,
    paths: Vec<PathBuf>,
}

impl<'plan> ShardSink<'plan> {
    fn new(out_dir: &'plan Path, shards: &'plan [MlxShardPlan]) -> Self {
        Self {
            out_dir,
            shards,
            next_shard: 0,
            open: None,
            active: None,
            held: Vec::new(),
            written_bytes: 0,
            paths: Vec::new(),
        }
    }

    fn held_bytes(&self) -> u64 {
        self.held.iter().map(|part| part.len() as u64).sum()
    }

    fn push(&mut self, chunk: WeightChunk) -> Result<(), InteropError> {
        let mut parts = chunk.parts.into_iter();
        if chunk.first {
            self.held.clear();
            self.start_tensor()?;
        }
        let Some(streamed) = parts.next() else {
            return Ok(());
        };
        self.write(&streamed)?;
        let rest: Vec<Vec<u8>> = parts.collect();
        self.held.resize_with(rest.len(), Vec::new);
        self.held
            .iter_mut()
            .zip(rest)
            .for_each(|(slot, part)| slot.extend_from_slice(&part));
        if chunk.last {
            self.flush_held()?;
        }
        Ok(())
    }

    fn flush_held(&mut self) -> Result<(), InteropError> {
        for part in core::mem::take(&mut self.held) {
            self.start_tensor()?;
            self.write(&part)?;
        }
        Ok(())
    }

    fn start_tensor(&mut self) -> Result<(), InteropError> {
        self.close_tensor()?;
        let exhausted = self
            .open
            .as_ref()
            .is_none_or(|open| open.pending.is_empty());
        if exhausted {
            self.close_shard()?;
            self.open_next_shard()?;
        }
        let unplanned = || InteropError::MlxWeightMismatch {
            tensor: "tensor beyond the planned layout".to_string(),
            expected: 0,
            found: 0,
        };
        let open = self.open.as_mut().ok_or_else(unplanned)?;
        let (header, rest) = open.pending.split_first().ok_or_else(unplanned)?;
        open.pending = rest;
        self.active = Some(ActiveTensor { header, written: 0 });
        Ok(())
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), InteropError> {
        let Some(open) = self.open.as_mut() else {
            return Ok(());
        };
        open.writer.write_all(bytes).map_err(io_error(&open.path))?;
        if let Some(active) = self.active.as_mut() {
            active.written += bytes.len() as u64;
        }
        self.written_bytes += bytes.len() as u64;
        Ok(())
    }

    fn close_tensor(&mut self) -> Result<(), InteropError> {
        let Some(active) = self.active.take() else {
            return Ok(());
        };
        let expected = active.header.byte_len();
        proxima_telemetry::debug!(
            tensor = ?active.header.name,
            tensor_bytes = active.written,
            held_bytes = self.held_bytes(),
            written_bytes = self.written_bytes,
            "mlx tensor written"
        );
        match active.written == expected {
            true => Ok(()),
            false => Err(InteropError::MlxWeightMismatch {
                tensor: active.header.name.clone(),
                expected,
                found: active.written,
            }),
        }
    }

    fn close_shard(&mut self) -> Result<(), InteropError> {
        let Some(mut open) = self.open.take() else {
            return Ok(());
        };
        if let Some(missing) = open.pending.first() {
            return Err(InteropError::MlxWeightMismatch {
                tensor: missing.name.clone(),
                expected: missing.byte_len(),
                found: 0,
            });
        }
        open.writer.flush().map_err(io_error(&open.path))?;
        proxima_telemetry::debug!(
            path = ?open.path,
            written_bytes = self.written_bytes,
            "mlx shard closed"
        );
        Ok(())
    }

    fn open_next_shard(&mut self) -> Result<(), InteropError> {
        let Some(shard) = self.shards.get(self.next_shard) else {
            return Ok(());
        };
        self.next_shard += 1;
        let path = self.out_dir.join(&shard.file_name);
        let file = File::create(&path).map_err(io_error(&path))?;
        let mut writer = BufWriter::with_capacity(WRITE_BUFFER_BYTES, file);
        writer.write_all(&shard.prefix).map_err(io_error(&path))?;
        self.paths.push(path.clone());
        self.written_bytes += shard.prefix.len() as u64;
        self.open = Some(OpenShard {
            path,
            writer,
            pending: &shard.tensors,
        });
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<PathBuf>, InteropError> {
        self.close_tensor()?;
        self.close_shard()?;
        while self.next_shard < self.shards.len() {
            self.open_next_shard()?;
            self.close_shard()?;
        }
        Ok(self.paths)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use proxima_gguf::value::MetadataValue;
    use proxima_gguf::{GgufModel, write_complete};

    use proxima_safetensors::TensorPayload;

    use super::*;
    use crate::transform::MlxExport;

    fn pattern(len: usize, seed: usize) -> Vec<u8> {
        (0..len)
            .map(|index| ((index * 31 + seed * 7) % 251) as u8)
            .collect()
    }

    fn chunk(unit: usize, position: (bool, bool), parts: &[&[u8]]) -> WeightChunk {
        WeightChunk {
            unit,
            first: position.0,
            last: position.1,
            parts: parts.iter().map(|part| part.to_vec()).collect(),
        }
    }

    #[test]
    fn streamed_shards_are_byte_equal_to_assembled_shards_across_a_triple_split_over_blocks() {
        let flat = pattern(16, 1);
        let weight = pattern(16, 2);
        let scales = pattern(4, 3);
        let biases = pattern(4, 4);
        let last = pattern(16, 5);
        let header = |name: &str, dtype, shape: &[u64]| TensorHeader {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
        };
        let headers = vec![
            header("a.weight", DType::BFLOAT16, &[2, 4]),
            header("b.weight", DType::UINT32, &[2, 2]),
            header("b.scales", DType::BFLOAT16, &[2, 1]),
            header("b.biases", DType::BFLOAT16, &[2, 1]),
            header("c.weight", DType::BFLOAT16, &[2, 4]),
        ];
        let payload = |header: &TensorHeader, data| TensorPayload {
            name: header.name.clone(),
            dtype: header.dtype,
            shape: header.shape.clone(),
            data,
        };
        let datas = [&flat, &weight, &scales, &biases, &last];
        let assembled = MlxExport::assemble(
            headers
                .iter()
                .zip(datas)
                .map(|(header, data)| payload(header, data))
                .collect(),
            20,
        )
        .expect("assembles");
        let layout = MlxLayout::plan(headers, 20).expect("plans");
        let scratch = tempfile::tempdir().expect("tempdir");
        let mut sink = ShardSink::new(scratch.path(), &layout.shards);

        sink.push(chunk(0, (true, false), &[&flat[..8]]))
            .expect("a block 0");
        sink.push(chunk(0, (false, true), &[&flat[8..]]))
            .expect("a block 1");
        sink.push(chunk(
            1,
            (true, false),
            &[&weight[..8], &scales[..2], &biases[..2]],
        ))
        .expect("b block 0");
        sink.push(chunk(
            1,
            (false, true),
            &[&weight[8..], &scales[2..], &biases[2..]],
        ))
        .expect("b block 1");
        sink.push(chunk(2, (true, true), &[&last]))
            .expect("c block");
        let paths = sink.finish().expect("finishes");

        assert!(
            assembled.shards.len() > 1,
            "the cap splits the triple across shards"
        );
        assert_eq!(paths.len(), assembled.shards.len());
        for (path, shard) in paths.iter().zip(&assembled.shards) {
            assert_eq!(
                path.file_name().and_then(|name| name.to_str()),
                Some(shard.file_name.as_str())
            );
            assert_eq!(
                std::fs::read(path).expect("read shard"),
                shard.bytes,
                "{}",
                shard.file_name
            );
        }
        assert_eq!(layout.index, assembled.index);
    }

    #[test]
    fn streaming_fewer_bytes_than_the_header_promised_names_the_tensor() {
        let header = TensorHeader {
            name: "a.weight".to_string(),
            dtype: DType::BFLOAT16,
            shape: vec![2, 4],
        };
        let layout = MlxLayout::plan(vec![header], MLX_MAX_SHARD_BYTES).expect("plans");
        let scratch = tempfile::tempdir().expect("tempdir");
        let mut sink = ShardSink::new(scratch.path(), &layout.shards);

        sink.push(chunk(0, (true, true), &[&pattern(15, 1)]))
            .expect("short block");
        let refusal = sink.finish().unwrap_err();

        assert!(matches!(
            refusal,
            InteropError::MlxWeightMismatch { tensor, expected: 16, found: 15 } if tensor == "a.weight"
        ));
    }

    fn gguf_of_family(family: &str) -> Vec<u8> {
        write_complete(&GgufModel {
            version: 3,
            metadata: vec![(
                "general.architecture".to_string(),
                MetadataValue::String(family.to_string()),
            )],
            tensors: Vec::new(),
        })
        .expect("writes the gguf")
    }

    #[test]
    fn mlx_directory_export_names_the_missing_gguf_in_the_io_error() {
        let scratch = tempfile::tempdir().expect("tempdir");
        let missing = scratch.path().join("absent.gguf");

        let refusal =
            export_mlx_directory(&missing, &scratch.path().join("out"), None, DType::BFLOAT16)
                .unwrap_err();

        assert!(matches!(refusal, InteropError::MlxDirectoryIo { path, .. } if path == missing));
    }

    #[test]
    fn mlx_directory_export_writes_nothing_when_the_family_is_refused() {
        let scratch = tempfile::tempdir().expect("tempdir");
        let source = scratch.path().join("llama.gguf");
        std::fs::write(&source, gguf_of_family("llama")).expect("write the gguf");
        let out = scratch.path().join("out");

        let refusal = export_mlx_directory(&source, &out, None, DType::BFLOAT16).unwrap_err();

        assert!(matches!(refusal, InteropError::UnknownFamilyConfig { .. }));
        assert!(!out.exists());
    }
}
