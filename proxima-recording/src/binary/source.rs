use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_stream::try_stream;
use serde_json::Value;

use crate::binary::frame::decode_block;
use crate::binary::wire::{FRAME_LEN_MASK, FRAME_STORED_FLAG};
use crate::factory::{RecordingSourceFactory, RecordingSourceRegistry, SourceBuildFuture};
use crate::rt_fs::offload;
use crate::source::{DynRecordingSource, RecordingEventStream, RecordingSource};
use proxima_core::ProximaError;
use proxima_runtime::Runtime;

#[cfg(feature = "durable-wal")]
use crate::event::RecordingEvent;
#[cfg(feature = "durable-wal")]
use futures::stream::Stream;
#[cfg(feature = "durable-wal")]
use std::io::{Seek, SeekFrom};
#[cfg(feature = "durable-wal")]
use std::pin::Pin;

/// Stream of (start_byte_offset, RecordingEvent) tuples. The offset is the
/// byte position of the frame-length prefix in the data file, suitable for
/// resuming a consumer via [`BinSource::events_from_offset`].
///
/// S1 of the proxima-notify initiative.
#[cfg(feature = "durable-wal")]
pub type OffsetEventStream<'lt> =
    Pin<Box<dyn Stream<Item = Result<(u64, RecordingEvent), ProximaError>> + Send + 'lt>>;

pub struct BinSource {
    path: PathBuf,
    runtime: Arc<dyn Runtime>,
}

/// One frame read off the wire: the decoded inner bytes plus the on-disk
/// block length (4-byte prefix excluded), or `None` at clean EOF. The block
/// length feeds the offset cursor; the wire format (4-byte masked length
/// prefix + stored/zstd block) is unchanged from the async path.
type FrameRead = Option<(Vec<u8>, usize)>;

// read one frame from `reader`, decompressing if needed. blocking; runs inside
// an offload closure so the calling core is yielded. returns None at clean EOF.
fn read_next_frame(reader: &mut BufReader<File>) -> Result<FrameRead, ProximaError> {
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
    Ok(Some((inner, block_len)))
}

impl BinSource {
    #[must_use]
    pub fn new(path: impl Into<PathBuf>, runtime: Arc<dyn Runtime>) -> Self {
        Self {
            path: path.into(),
            runtime,
        }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Open a stream that begins at `start_offset` bytes into the data
    /// file and yields `(frame_start_offset, RecordingEvent)` tuples until
    /// EOF. The frame_start_offset for each yielded event is the byte
    /// position of that frame's length prefix — exactly what a consumer
    /// commits via the offset cursor (C7) AFTER processing the event.
    ///
    /// A `start_offset` of `0` is equivalent to `events()` but with offset
    /// pairing. If `start_offset` lands inside a frame (the consumer
    /// committed a corrupt position), this method returns an error on the
    /// first yield. Consumers MUST commit offsets at frame boundaries
    /// (which the offset returned by this stream guarantees).
    ///
    /// Does NOT live-tail past EOF — a follow-up enhancement (a
    /// file-watcher loop) is out of scope for S1. Consumers wanting
    /// live-tail call this method, drain to EOF, then sleep + retry from
    /// the last-yielded offset.
    ///
    /// S1 of the proxima-notify initiative.
    #[cfg(feature = "durable-wal")]
    #[must_use]
    pub fn events_from_offset<'lt>(&'lt self, start_offset: u64) -> OffsetEventStream<'lt> {
        let path = self.path.clone();
        let runtime = Arc::clone(&self.runtime);
        let stream = try_stream! {
            let open_path = path.clone();
            let reader = offload(&runtime, move || {
                let mut file = File::open(&open_path)
                    .map_err(|err| ProximaError::Record(format!("open bin source: {err}")))?;
                file.seek(SeekFrom::Start(start_offset))
                    .map_err(|err| ProximaError::Record(format!("seek bin source to {start_offset}: {err}")))?;
                Ok(BufReader::new(file))
            }).await?;
            let reader = Arc::new(Mutex::new(reader));
            let mut next_offset = start_offset;
            loop {
                let frame_start = next_offset;
                let reader = Arc::clone(&reader);
                let frame = offload(&runtime, move || {
                    let mut guard = reader
                        .lock()
                        .map_err(|err| ProximaError::Record(format!("bin source poisoned: {err}")))?;
                    read_next_frame(&mut guard)
                }).await?;
                let Some((inner, block_len)) = frame else { break };
                // advance: 4-byte length prefix + on-disk block bytes. a block
                // holds N events sharing this block's start offset; resume is
                // block-granular (a compressed block can't be seeked into).
                next_offset = frame_start + 4 + block_len as u64;
                for event in decode_block(&inner)? {
                    yield (frame_start, event);
                }
            }
        };
        Box::pin(stream)
    }
}

impl RecordingSource for BinSource {
    fn events<'lifetime>(&'lifetime self) -> RecordingEventStream<'lifetime> {
        let path = self.path.clone();
        let runtime = Arc::clone(&self.runtime);
        let stream = try_stream! {
            let open_path = path.clone();
            let reader = offload(&runtime, move || {
                File::open(&open_path)
                    .map(BufReader::new)
                    .map_err(|err| ProximaError::Record(format!("open bin source: {err}")))
            }).await?;
            let reader = Arc::new(Mutex::new(reader));
            loop {
                let reader = Arc::clone(&reader);
                let frame = offload(&runtime, move || {
                    let mut guard = reader
                        .lock()
                        .map_err(|err| ProximaError::Record(format!("bin source poisoned: {err}")))?;
                    read_next_frame(&mut guard)
                }).await?;
                let Some((inner, _block_len)) = frame else { break };
                for event in decode_block(&inner)? {
                    yield event;
                }
            }
        };
        Box::pin(stream)
    }
}

pub struct BinSourceFactory;

impl RecordingSourceFactory for BinSourceFactory {
    fn name(&self) -> &str {
        "bin"
    }

    fn build<'lifetime>(
        &'lifetime self,
        spec: &'lifetime Value,
        registry: &'lifetime RecordingSourceRegistry,
    ) -> SourceBuildFuture<'lifetime> {
        Box::pin(async move {
            let path = spec
                .get("path")
                .or_else(|| spec.get("source"))
                .and_then(Value::as_str)
                .ok_or_else(|| ProximaError::Config("bin source requires `path`".into()))?
                .to_string();
            let runtime = registry.runtime()?;
            let dyn_source: DynRecordingSource = Arc::new(BinSource::new(path, runtime));
            Ok(dyn_source)
        })
    }
}
