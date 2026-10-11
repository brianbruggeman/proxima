use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use proxima_gguf::{GgmlType, ParsedGguf, TensorInfo, parse_complete};

use crate::error::RecordError;

const HEADER_CAPS: [usize; 5] = [4 << 20, 16 << 20, 64 << 20, 128 << 20, 512 << 20];

pub struct Checkpoint {
    path: PathBuf,
    parsed: ParsedGguf,
}

pub struct Rows {
    pub bytes: Vec<u8>,
    pub tensor: String,
    pub byte_offset: u64,
}

fn parse_header(path: &Path) -> Result<ParsedGguf, RecordError> {
    let mut file = File::open(path).map_err(RecordError::io(path))?;
    for cap in HEADER_CAPS {
        let mut buffer = vec![0_u8; cap];
        file.seek(SeekFrom::Start(0))
            .map_err(RecordError::io(path))?;
        let read = file.read(&mut buffer).map_err(RecordError::io(path))?;
        buffer.truncate(read);
        if let Ok(parsed) = parse_complete(&buffer) {
            return Ok(parsed);
        }
    }
    Err(RecordError::Header {
        path: path.to_path_buf(),
    })
}

impl Checkpoint {
    pub fn open(path: &Path) -> Result<Self, RecordError> {
        Ok(Self {
            path: path.to_path_buf(),
            parsed: parse_header(path)?,
        })
    }

    pub fn blob_name(&self) -> String {
        self.path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    pub fn tensor_of(
        &self,
        kind: GgmlType,
        fragment: &str,
        minimum: usize,
    ) -> Result<&TensorInfo, RecordError> {
        self.parsed
            .tensors
            .iter()
            .find(|tensor| {
                tensor.ggml_type == kind
                    && tensor.name.contains(fragment)
                    && tensor.element_count() as usize >= minimum
            })
            .ok_or_else(|| RecordError::NoTensor {
                path: self.path.clone(),
                kind: format!("{kind:?}"),
                fragment: fragment.to_string(),
            })
    }

    pub fn read_rows(&self, tensor: &TensorInfo, byte_len: usize) -> Result<Rows, RecordError> {
        let byte_offset = self.parsed.data_offset + tensor.offset;
        let mut file = File::open(&self.path).map_err(RecordError::io(&self.path))?;
        file.seek(SeekFrom::Start(byte_offset))
            .map_err(RecordError::io(&self.path))?;
        let mut bytes = vec![0_u8; byte_len];
        file.read_exact(&mut bytes)
            .map_err(|_| RecordError::Short {
                tensor: tensor.name.clone(),
                needed: byte_len,
            })?;
        Ok(Rows {
            bytes,
            tensor: tensor.name.clone(),
            byte_offset,
        })
    }
}
