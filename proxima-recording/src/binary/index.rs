use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::event::InteractionId;
use crate::rt_fs::offload;
use proxima_core::ProximaError;
use proxima_runtime::Runtime;

pub const INDEX_RECORD_BYTES: u64 = 36;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexRecord {
    pub entry_offset: u64,
    pub ts_ms: u64,
    pub frame_len: u32,
    pub id: InteractionId,
}

impl IndexRecord {
    pub(crate) fn to_bytes(self) -> [u8; 36] {
        let mut buffer = [0_u8; 36];
        buffer[0..8].copy_from_slice(&self.entry_offset.to_le_bytes());
        buffer[8..16].copy_from_slice(&self.ts_ms.to_le_bytes());
        buffer[16..20].copy_from_slice(&self.frame_len.to_le_bytes());
        buffer[20..36].copy_from_slice(&self.id.to_bytes());
        buffer
    }

    fn from_bytes(buffer: [u8; 36]) -> Self {
        let mut offset_bytes = [0_u8; 8];
        offset_bytes.copy_from_slice(&buffer[0..8]);
        let mut ts_bytes = [0_u8; 8];
        ts_bytes.copy_from_slice(&buffer[8..16]);
        let mut len_bytes = [0_u8; 4];
        len_bytes.copy_from_slice(&buffer[16..20]);
        let mut id_bytes = [0_u8; 16];
        id_bytes.copy_from_slice(&buffer[20..36]);
        Self {
            entry_offset: u64::from_le_bytes(offset_bytes),
            ts_ms: u64::from_le_bytes(ts_bytes),
            frame_len: u32::from_le_bytes(len_bytes),
            id: InteractionId::from_bytes(id_bytes),
        }
    }
}

pub struct IndexWriter {
    path: PathBuf,
    // WHY Mutex here / WHY NOT removable / WHY right: same pattern
    // as `AppendLog` (src/pipe/log_pipe.rs, the `pipe` feature) — file
    // writes need exclusive access for record-atomic index entry
    // writes (36-byte fixed-size records). The guard is only held
    // inside the offloaded blocking closure (a std::sync::Mutex), the
    // I/O itself runs on the runtime's blocking pool. Per-core file
    // sharding changes the index format; not warranted today.
    file: Arc<Mutex<File>>,
    runtime: Arc<dyn Runtime>,
}

impl IndexWriter {
    pub async fn create(
        path: impl Into<PathBuf>,
        runtime: Arc<dyn Runtime>,
    ) -> Result<Self, ProximaError> {
        let path = path.into();
        let open_path = path.clone();
        let file = offload(&runtime, move || {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&open_path)
                .map_err(|err| ProximaError::Record(format!("open bin idx: {err}")))
        })
        .await?;
        Ok(Self {
            path,
            file: Arc::new(Mutex::new(file)),
            runtime,
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub async fn append(&self, record: IndexRecord) -> Result<(), ProximaError> {
        let buffer = record.to_bytes();
        let file = Arc::clone(&self.file);
        offload(&self.runtime, move || {
            let mut guard = file
                .lock()
                .map_err(|err| ProximaError::Record(format!("bin idx poisoned: {err}")))?;
            guard
                .write_all(&buffer)
                .map_err(|err| ProximaError::Record(format!("append bin idx: {err}")))
        })
        .await
    }

    /// Append a pre-packed run of 36-byte records in one write. The caller
    /// (a [`crate::binary::FrameEncoder`] batch) guarantees the bytes are a
    /// whole number of `INDEX_RECORD_BYTES` records; a partial tail would
    /// desync the fixed-stride reader.
    pub async fn append_bytes(&self, records: &[u8]) -> Result<(), ProximaError> {
        debug_assert_eq!(records.len() as u64 % INDEX_RECORD_BYTES, 0);
        if records.is_empty() {
            return Ok(());
        }
        let records = records.to_vec();
        let file = Arc::clone(&self.file);
        offload(&self.runtime, move || {
            let mut guard = file
                .lock()
                .map_err(|err| ProximaError::Record(format!("bin idx poisoned: {err}")))?;
            guard
                .write_all(&records)
                .map_err(|err| ProximaError::Record(format!("append bin idx batch: {err}")))
        })
        .await
    }

    pub async fn flush(&self) -> Result<(), ProximaError> {
        let file = Arc::clone(&self.file);
        offload(&self.runtime, move || {
            let mut guard = file
                .lock()
                .map_err(|err| ProximaError::Record(format!("bin idx poisoned: {err}")))?;
            guard
                .flush()
                .map_err(|err| ProximaError::Record(format!("flush bin idx: {err}")))
        })
        .await
    }

    /// Flush and fsync the index file. Unlike [`Self::flush`] (which only
    /// drains the userspace buffer to the OS), this calls `sync_all` to
    /// push the OS buffer through to the disk device. Required for a
    /// crash-safe WAL — the data file should be `sync_all`'d FIRST so the
    /// idx never references bytes that aren't durable yet.
    ///
    /// S1 of the proxima-notify initiative.
    #[cfg(feature = "durable-wal")]
    pub async fn sync_now(&self) -> Result<(), ProximaError> {
        let file = Arc::clone(&self.file);
        offload(&self.runtime, move || {
            let mut guard = file
                .lock()
                .map_err(|err| ProximaError::Record(format!("bin idx poisoned: {err}")))?;
            guard
                .flush()
                .map_err(|err| ProximaError::Record(format!("flush bin idx for sync: {err}")))?;
            guard
                .sync_all()
                .map_err(|err| ProximaError::Record(format!("fsync bin idx: {err}")))
        })
        .await
    }
}

pub struct IndexReader {
    path: PathBuf,
    runtime: Arc<dyn Runtime>,
}

impl IndexReader {
    #[must_use]
    pub fn new(path: impl Into<PathBuf>, runtime: Arc<dyn Runtime>) -> Self {
        Self {
            path: path.into(),
            runtime,
        }
    }

    pub async fn record_count(&self) -> Result<u64, ProximaError> {
        let path = self.path.clone();
        let len = offload(&self.runtime, move || {
            std::fs::metadata(&path)
                .map(|metadata| metadata.len())
                .map_err(|err| ProximaError::Record(format!("stat bin idx: {err}")))
        })
        .await?;
        Ok(len / INDEX_RECORD_BYTES)
    }

    pub async fn read_at(&self, index: u64) -> Result<IndexRecord, ProximaError> {
        let path = self.path.clone();
        let buffer = offload(&self.runtime, move || {
            let mut file = File::open(&path)
                .map_err(|err| ProximaError::Record(format!("open bin idx: {err}")))?;
            file.seek(SeekFrom::Start(index * INDEX_RECORD_BYTES))
                .map_err(|err| ProximaError::Record(format!("seek bin idx: {err}")))?;
            let mut buffer = [0_u8; 36];
            file.read_exact(&mut buffer)
                .map_err(|err| ProximaError::Record(format!("read bin idx record: {err}")))?;
            Ok(buffer)
        })
        .await?;
        Ok(IndexRecord::from_bytes(buffer))
    }

    /// Smallest record-index whose ts_ms >= `target`. Returns `None` when no
    /// record satisfies the predicate (i.e. the target is past the last entry).
    /// Records must be appended in monotonically non-decreasing ts_ms order
    /// (the natural state when a sink writes per-interaction events as they
    /// flow). The binary search is O(log n).
    pub async fn seek_by_ts(&self, target: u64) -> Result<Option<u64>, ProximaError> {
        let count = self.record_count().await?;
        if count == 0 {
            return Ok(None);
        }
        let mut low = 0_u64;
        let mut high = count;
        while low < high {
            let mid = low + (high - low) / 2;
            let record = self.read_at(mid).await?;
            if record.ts_ms < target {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if low == count {
            Ok(None)
        } else {
            Ok(Some(low))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use prime::os::runtime::PrimeRuntime;
    use rstest::rstest;
    use tempfile::tempdir;

    fn prime() -> Arc<dyn Runtime> {
        Arc::new(PrimeRuntime::new(1).expect("prime"))
    }

    fn make_record(offset: u64, ts_ms: u64) -> IndexRecord {
        IndexRecord {
            entry_offset: offset,
            ts_ms,
            frame_len: 64,
            id: InteractionId::from_bytes([(ts_ms % 256) as u8; 16]),
        }
    }

    #[test]
    fn record_round_trips_through_bytes() {
        let record = make_record(123, 456);
        let restored = IndexRecord::from_bytes(record.to_bytes());
        assert_eq!(record, restored);
    }

    #[proxima::test]
    async fn writer_then_reader_returns_same_records() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("test.idx");
        let runtime = prime();
        let writer = IndexWriter::create(&path, Arc::clone(&runtime))
            .await
            .expect("create writer");
        let original = [
            make_record(0, 10),
            make_record(80, 25),
            make_record(160, 40),
        ];
        for record in original {
            writer.append(record).await.expect("append");
        }
        writer.flush().await.expect("flush");
        let reader = IndexReader::new(&path, runtime);
        assert_eq!(reader.record_count().await.expect("count"), 3);
        for (index, expected) in original.iter().enumerate() {
            let got = reader.read_at(index as u64).await.expect("read");
            assert_eq!(&got, expected);
        }
    }

    #[rstest]
    #[case::below_first(5, Some(0))]
    #[case::exact_first(10, Some(0))]
    #[case::between(20, Some(1))]
    #[case::exact_last(40, Some(2))]
    #[case::above_last(100, None)]
    #[proxima::test]
    async fn seek_by_ts_returns_first_record_with_ts_ge_target(
        #[case] target: u64,
        #[case] expected: Option<u64>,
    ) {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("seek.idx");
        let runtime = prime();
        let writer = IndexWriter::create(&path, Arc::clone(&runtime))
            .await
            .expect("create writer");
        for record in [
            make_record(0, 10),
            make_record(80, 25),
            make_record(160, 40),
        ] {
            writer.append(record).await.expect("append");
        }
        writer.flush().await.expect("flush");
        let reader = IndexReader::new(&path, runtime);
        let found = reader.seek_by_ts(target).await.expect("seek");
        assert_eq!(found, expected);
    }

    #[proxima::test]
    async fn seek_by_ts_on_empty_index_returns_none() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("empty.idx");
        let runtime = prime();
        let writer = IndexWriter::create(&path, Arc::clone(&runtime))
            .await
            .expect("create writer");
        writer.flush().await.expect("flush");
        let reader = IndexReader::new(&path, runtime);
        let found = reader.seek_by_ts(0).await.expect("seek");
        assert!(found.is_none());
    }
}
