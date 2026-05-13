//! `BinFormat` — the binary recording codec: `[u32 len|flag][zstd-or-stored
//! block of postcard frames]`. Pure codec (no file I/O, no index, no runtime);
//! the `AppendLog` Pipe owns the file and offset. The data-file bytes are
//! identical to the legacy `BinSink`'s data file, so old recordings read back
//! unchanged.

use std::io::BufRead;

use crate::binary::frame::{FrameEncoder, decode_block as decode_frames};
use crate::binary::wire::{FRAME_LEN_MASK, FRAME_STORED_FLAG};
use crate::event::RecordingEvent;
use crate::format::Format;
use proxima_core::ProximaError;

const ZSTD_DEFAULT_LEVEL: i32 = 3;
// blocks >= this compress as one unit; smaller store raw (high bit flags it).
// matches the legacy BinSink threshold — the data bytes stay identical.
const ZSTD_MIN_BLOCK_BYTES: usize = 256;
const BLOCK_LEN_PREFIX: u64 = 4;

/// The binary block codec. Holds a reused frame encoder + a persistent zstd
/// compressor so a streaming interaction allocates nothing per event and pays
/// the zstd init cost once.
pub struct BinFormat {
    encoder: FrameEncoder,
    compressor: zstd::bulk::Compressor<'static>,
}

impl BinFormat {
    pub fn new() -> Result<Self, ProximaError> {
        Self::with_level(ZSTD_DEFAULT_LEVEL)
    }

    pub fn with_level(zstd_level: i32) -> Result<Self, ProximaError> {
        let compressor = zstd::bulk::Compressor::new(zstd_level)
            .map_err(|err| ProximaError::Record(format!("zstd compressor init: {err}")))?;
        Ok(Self {
            encoder: FrameEncoder::new(),
            compressor,
        })
    }
}

impl Format for BinFormat {
    fn name(&self) -> &'static str {
        "bin"
    }

    fn encode_block(&mut self, events: Vec<RecordingEvent>) -> Result<Vec<u8>, ProximaError> {
        self.encoder.reset();
        for event in events {
            self.encoder.push(event)?;
        }
        let inner = self.encoder.inner();

        let compressed_holder;
        let (block, stored_flag): (&[u8], u32) = if inner.len() >= ZSTD_MIN_BLOCK_BYTES {
            compressed_holder = self
                .compressor
                .compress(inner)
                .map_err(|err| ProximaError::Record(format!("zstd compress block: {err}")))?;
            (&compressed_holder, 0)
        } else {
            (inner, FRAME_STORED_FLAG)
        };

        let block_len = u32::try_from(block.len())
            .ok()
            .filter(|len| *len <= FRAME_LEN_MASK)
            .ok_or_else(|| ProximaError::Record("bin block exceeds 2 GiB".into()))?;

        let mut frame = Vec::with_capacity(BLOCK_LEN_PREFIX as usize + block.len());
        frame.extend_from_slice(&(block_len | stored_flag).to_le_bytes());
        frame.extend_from_slice(block);
        Ok(frame)
    }

    fn decode_block(
        &self,
        reader: &mut dyn BufRead,
    ) -> Result<Option<(Vec<RecordingEvent>, u64)>, ProximaError> {
        let mut len_buffer = [0_u8; 4];
        match reader.read_exact(&mut len_buffer) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(err) => return Err(ProximaError::Record(format!("read bin frame len: {err}"))),
        }
        let raw_len = u32::from_le_bytes(len_buffer);
        let stored = raw_len & FRAME_STORED_FLAG != 0;
        let block_len = (raw_len & FRAME_LEN_MASK) as usize;

        let mut block = vec![0_u8; block_len];
        reader
            .read_exact(&mut block)
            .map_err(|err| ProximaError::Record(format!("read bin block: {err}")))?;

        let inner = if stored {
            block
        } else {
            zstd::stream::decode_all(block.as_slice())
                .map_err(|err| ProximaError::Record(format!("zstd decompress block: {err}")))?
        };
        let events = decode_frames(&inner)?;
        Ok(Some((events, BLOCK_LEN_PREFIX + block_len as u64)))
    }
}
