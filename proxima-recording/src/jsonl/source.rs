use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_stream::try_stream;
use serde_json::Value;

use crate::factory::{RecordingSourceFactory, RecordingSourceRegistry, SourceBuildFuture};
use crate::jsonl::wire::{WireEnvelope, envelope_to_event};
use crate::rt_fs::offload;
use crate::source::{DynRecordingSource, RecordingEventStream, RecordingSource};
use proxima_core::ProximaError;
use proxima_runtime::Runtime;

pub struct JsonlSource {
    path: PathBuf,
    runtime: Arc<dyn Runtime>,
}

// read one line (newline-terminated) off `reader`, or None at EOF. The
// trailing newline is stripped to match tokio's `Lines::next_line`.
fn read_next_line(reader: &mut BufReader<File>) -> Result<Option<String>, ProximaError> {
    let mut line = String::new();
    let read = reader
        .read_line(&mut line)
        .map_err(|err| ProximaError::Record(format!("read jsonl line: {err}")))?;
    if read == 0 {
        return Ok(None);
    }
    if line.ends_with('\n') {
        line.pop();
        if line.ends_with('\r') {
            line.pop();
        }
    }
    Ok(Some(line))
}

impl JsonlSource {
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
}

pub struct JsonlSourceFactory;

impl RecordingSourceFactory for JsonlSourceFactory {
    fn name(&self) -> &str {
        "jsonl"
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
                .ok_or_else(|| ProximaError::Config("jsonl source requires `path`".into()))?
                .to_string();
            let runtime = registry.runtime()?;
            let dyn_source: DynRecordingSource = Arc::new(JsonlSource::new(path, runtime));
            Ok(dyn_source)
        })
    }
}

impl RecordingSource for JsonlSource {
    fn events<'lifetime>(&'lifetime self) -> RecordingEventStream<'lifetime> {
        let path = self.path.clone();
        let runtime = Arc::clone(&self.runtime);
        let stream = try_stream! {
            let open_path = path.clone();
            let reader = offload(&runtime, move || {
                File::open(&open_path)
                    .map(BufReader::new)
                    .map_err(|err| ProximaError::Record(format!("open jsonl source: {err}")))
            }).await?;
            let reader = Arc::new(Mutex::new(reader));
            loop {
                let reader = Arc::clone(&reader);
                let line = offload(&runtime, move || {
                    let mut guard = reader
                        .lock()
                        .map_err(|err| ProximaError::Record(format!("jsonl source poisoned: {err}")))?;
                    read_next_line(&mut guard)
                }).await?;
                let Some(line) = line else { break };
                if line.trim().is_empty() {
                    continue;
                }
                let envelope: WireEnvelope = serde_json::from_str(&line)
                    .map_err(|err| ProximaError::Record(format!("parse jsonl line: {err}")))?;
                let event = envelope_to_event(envelope)?;
                yield event;
            }
        };
        Box::pin(stream)
    }
}
