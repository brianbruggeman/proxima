//! `AppendLog` / `ReplayLog` — the durable recording terminal as typed Pipes.
//!
//! The sink IS a Pipe (no bespoke `RecordingSink`). With the generic
//! `Pipe<In, Out>` trait the terminals are typed end-to-end — no method-byte
//! dispatch, no `Carry` erasure:
//!
//! - `AppendLog: Pipe<Vec<RecordingEvent>, AppendAck>` — the write terminal:
//!   `call(events) -> AppendAck{offset}`. Durability control (`flush`/`sync`)
//!   is not a data-plane transform, so it lives in inherent methods rather
//!   than multiplexed into the Pipe input.
//! - `ReplayLog: Pipe<u64, ReplayChunk>` — the read terminal:
//!   `call(cursor) -> ReplayChunk{events, next_offset, done}`, cursor-paginated
//!   so a huge recording streams block-by-block.
//!
//! Both hold a `Box<dyn Format>` (the format is a config-selected CODEC axis,
//! not a peer type) + the file, and run their blocking I/O off the caller's
//! core via the `rt_fs` offload, on any injected `Runtime`. Fan-out to N
//! durables-each-with-its-own-format is composition over these (`FanOut`).

use core::future::Future;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::event::RecordingEvent;
use crate::format::Format;
use crate::rt_fs::offload;
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_runtime::Runtime;

/// `call` ack: the byte offset the block landed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppendAck {
    pub offset: u64,
}

/// One replay step: the unit's events, the cursor for the next call, and EOF.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayChunk {
    pub events: Vec<RecordingEvent>,
    pub next_offset: u64,
    pub done: bool,
}

struct WriteState {
    file: File,
    offset: u64,
    format: Box<dyn Format>,
}

/// Write terminal: append batches durably, format-selected, runtime-injected.
pub struct AppendLog {
    state: Arc<Mutex<WriteState>>,
    runtime: Arc<dyn Runtime>,
}

impl AppendLog {
    /// Open (create-or-append) the destination with the given format codec.
    pub fn open(
        path: impl Into<PathBuf>,
        format: Box<dyn Format>,
        runtime: Arc<dyn Runtime>,
    ) -> Result<Self, ProximaError> {
        let path = path.into();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(false)
            .open(&path)
            .map_err(|err| ProximaError::Record(format!("open append-log: {err}")))?;
        let offset = file.metadata().map(|meta| meta.len()).unwrap_or(0);
        Ok(Self {
            state: Arc::new(Mutex::new(WriteState {
                file,
                offset,
                format,
            })),
            runtime,
        })
    }

    /// Flush buffered writes to the OS. Durability control, not a transform.
    pub async fn flush(&self) -> Result<(), ProximaError> {
        let state = Arc::clone(&self.state);
        offload(&self.runtime, move || {
            let mut guard = state
                .lock()
                .map_err(|err| ProximaError::Record(format!("log poisoned: {err}")))?;
            guard
                .file
                .flush()
                .map_err(|err| ProximaError::Record(format!("flush: {err}")))
        })
        .await
    }

    /// Flush then fsync — the block is on stable storage when this returns.
    pub async fn sync(&self) -> Result<(), ProximaError> {
        let state = Arc::clone(&self.state);
        offload(&self.runtime, move || {
            let mut guard = state
                .lock()
                .map_err(|err| ProximaError::Record(format!("log poisoned: {err}")))?;
            guard
                .file
                .flush()
                .map_err(|err| ProximaError::Record(format!("flush for sync: {err}")))?;
            guard
                .file
                .sync_all()
                .map_err(|err| ProximaError::Record(format!("fsync: {err}")))
        })
        .await
    }
}

impl SendPipe for AppendLog {
    type In = Vec<RecordingEvent>;
    type Out = AppendAck;
    type Err = ProximaError;

    fn call(
        &self,
        events: Vec<RecordingEvent>,
    ) -> impl Future<Output = Result<AppendAck, ProximaError>> + Send {
        let runtime = Arc::clone(&self.runtime);
        let state = Arc::clone(&self.state);
        async move {
            let offset = offload(&runtime, move || {
                let mut guard = state
                    .lock()
                    .map_err(|err| ProximaError::Record(format!("log poisoned: {err}")))?;
                let write = &mut *guard;
                let bytes = write.format.encode_block(events)?;
                write
                    .file
                    .write_all(&bytes)
                    .map_err(|err| ProximaError::Record(format!("write block: {err}")))?;
                let at = write.offset;
                write.offset += bytes.len() as u64;
                Ok(at)
            })
            .await?;
            Ok(AppendAck { offset })
        }
    }
}

struct ReadState {
    file: File,
    format: Box<dyn Format>,
}

/// Read terminal: cursor-paginated replay, format-selected, runtime-injected.
pub struct ReplayLog {
    state: Arc<Mutex<ReadState>>,
    runtime: Arc<dyn Runtime>,
}

impl ReplayLog {
    pub fn open(
        path: impl Into<PathBuf>,
        format: Box<dyn Format>,
        runtime: Arc<dyn Runtime>,
    ) -> Result<Self, ProximaError> {
        let file = File::open(path.into())
            .map_err(|err| ProximaError::Record(format!("open replay-log: {err}")))?;
        Ok(Self {
            state: Arc::new(Mutex::new(ReadState { file, format })),
            runtime,
        })
    }
}

impl SendPipe for ReplayLog {
    type In = u64;
    type Out = ReplayChunk;
    type Err = ProximaError;

    fn call(
        &self,
        from_offset: u64,
    ) -> impl Future<Output = Result<ReplayChunk, ProximaError>> + Send {
        let runtime = Arc::clone(&self.runtime);
        let state = Arc::clone(&self.state);
        async move {
            offload(&runtime, move || {
                let mut guard = state
                    .lock()
                    .map_err(|err| ProximaError::Record(format!("log poisoned: {err}")))?;
                let read = &mut *guard;
                read.file
                    .seek(SeekFrom::Start(from_offset))
                    .map_err(|err| ProximaError::Record(format!("seek replay: {err}")))?;
                // fresh BufReader per call: its read-ahead is discarded, so the
                // next call re-seeks cleanly — offset tracked via bytes consumed.
                let mut reader = BufReader::new(&mut read.file);
                match read.format.decode_block(&mut reader)? {
                    Some((events, consumed)) => Ok(ReplayChunk {
                        events,
                        next_offset: from_offset + consumed,
                        done: false,
                    }),
                    None => Ok(ReplayChunk {
                        events: Vec::new(),
                        next_offset: from_offset,
                        done: true,
                    }),
                }
            })
            .await
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(crate) mod test_support {
    use super::*;
    use crate::event::{FrameMetadata, HttpEvent, InteractionId, ProtocolEvent};
    use bytes::Bytes;

    pub(crate) fn sample_events() -> Vec<RecordingEvent> {
        let id = InteractionId::new();
        let mut events = vec![event(id, b"data: small\n\n")];
        events.extend(
            (0..32).map(|_| event(id, b"data: {\"delta\":\"hello there from the stream\"}\n\n")),
        );
        events
    }

    fn event(id: InteractionId, data: &'static [u8]) -> RecordingEvent {
        RecordingEvent {
            id,
            ts_ms: 7,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::ResponseChunk {
                data: Bytes::from_static(data),
                metadata: FrameMetadata::new(),
            }),
        }
    }

    // drive append+flush through the write pipe, then paginate replay through
    // the read pipe; return the reassembled events.
    pub(crate) async fn append_then_replay(
        writer: &AppendLog,
        reader: &ReplayLog,
        events: Vec<RecordingEvent>,
    ) -> Vec<RecordingEvent> {
        writer.call(events).await.unwrap();
        writer.flush().await.unwrap();
        drain(reader).await
    }

    pub(crate) async fn drain(reader: &ReplayLog) -> Vec<RecordingEvent> {
        let mut replayed = Vec::new();
        let mut offset = 0_u64;
        loop {
            let chunk = reader.call(offset).await.unwrap();
            if chunk.done {
                break;
            }
            replayed.extend(chunk.events);
            offset = chunk.next_offset;
        }
        replayed
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::test_support::{append_then_replay, drain, sample_events};
    use super::*;
    use crate::{BinFormat, JsonFormat};
    use prime::os::runtime::PrimeRuntime;
    use proxima_runtime::tokio::TokioPerCoreRuntime;

    fn prime() -> Arc<dyn Runtime> {
        Arc::new(PrimeRuntime::new(1).expect("prime"))
    }

    // bin round-trips through the typed write/read pipes on prime.
    #[test]
    fn bin_round_trips_on_prime() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rec.bin");
        let runtime = prime();
        let written = sample_events();
        let replayed = futures::executor::block_on(async {
            let writer = AppendLog::open(
                &path,
                Box::new(BinFormat::new().unwrap()),
                Arc::clone(&runtime),
            )
            .unwrap();
            let reader =
                ReplayLog::open(&path, Box::new(BinFormat::new().unwrap()), runtime).unwrap();
            append_then_replay(&writer, &reader, written.clone()).await
        });
        assert_eq!(replayed, written, "bin round-trips on prime");
    }

    // json round-trips through the SAME typed pipes — only the format codec varies.
    #[test]
    fn json_round_trips_on_prime() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rec.jsonl");
        let runtime = prime();
        let written = sample_events();
        let replayed = futures::executor::block_on(async {
            let writer =
                AppendLog::open(&path, Box::new(JsonFormat::new()), Arc::clone(&runtime)).unwrap();
            let reader = ReplayLog::open(&path, Box::new(JsonFormat::new()), runtime).unwrap();
            append_then_replay(&writer, &reader, written.clone()).await
        });
        assert_eq!(replayed, written, "json round-trips on prime");
    }

    // runtime swap: bin output is byte-identical whether prime or tokio drives.
    #[test]
    fn bin_bytes_identical_prime_vs_tokio() {
        let dir = tempfile::tempdir().unwrap();
        let events = sample_events();

        let prime_path = dir.path().join("prime.bin");
        let prime_rt = prime();
        futures::executor::block_on(async {
            let writer =
                AppendLog::open(&prime_path, Box::new(BinFormat::new().unwrap()), prime_rt)
                    .unwrap();
            writer.call(events.clone()).await.unwrap();
            writer.flush().await.unwrap();
        });

        let tokio_path = dir.path().join("tokio.bin");
        let tokio_rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let events_for_tokio = events.clone();
        tokio_rt.block_on(async {
            let runtime: Arc<dyn Runtime> = Arc::new(TokioPerCoreRuntime::new(1).expect("tokio"));
            let writer =
                AppendLog::open(&tokio_path, Box::new(BinFormat::new().unwrap()), runtime).unwrap();
            writer.call(events_for_tokio).await.unwrap();
            writer.flush().await.unwrap();
        });

        assert_eq!(
            std::fs::read(&prime_path).unwrap(),
            std::fs::read(&tokio_path).unwrap(),
            "bin bytes identical across runtimes"
        );
    }

    // a replay against an empty log is immediately done, no events.
    #[test]
    fn empty_log_replays_done() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        let runtime = prime();
        let replayed = futures::executor::block_on(async {
            AppendLog::open(
                &path,
                Box::new(BinFormat::new().unwrap()),
                Arc::clone(&runtime),
            )
            .unwrap();
            let reader =
                ReplayLog::open(&path, Box::new(BinFormat::new().unwrap()), runtime).unwrap();
            drain(&reader).await
        });
        assert!(replayed.is_empty(), "empty log yields no events");
    }
}
