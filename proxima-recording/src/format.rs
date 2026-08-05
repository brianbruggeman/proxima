//! `Format` — the codec axis of the durable log (the "schema", not a "logger").
//!
//! A `Format` turns a batch of events into the bytes appended to the log, and
//! decodes one unit back. It owns FRAMING (bin = `[u32 len|flag]` + zstd block;
//! json = one line per event) because bin's block-level compression spans a
//! whole batch — but it does NO file I/O and holds no runtime. The ONE durable
//! terminal Pipe (`crate::pipe::AppendLog`, the `pipe` feature) owns the file, the
//! offset, and the `rt_fs` offload; the format is a config-selected field on
//! it. bin/json are `Format` impls, not peer "writer" types — adding a format
//! is one impl + one registry line, and choosing one is config.

use crate::event::RecordingEvent;
use proxima_core::ProximaError;
use std::io::BufRead;

/// Codec for one on-disk recording format. `Send` so the terminal Pipe can
/// move it into a background-pool offload closure.
///
/// Encode and decode are exact inverses over a batch — the property every
/// durable terminal above this trait relies on:
///
/// ```
/// use proxima_recording::{
///     BinFormat, Format, HttpEvent, InteractionId, ProtocolEvent, RecordingEvent,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let event = RecordingEvent {
///     id: InteractionId::from_bytes([7; 16]),
///     ts_ms: 42,
///     parent: None,
///     event: ProtocolEvent::Http(HttpEvent::RequestEnded),
/// };
///
/// let mut codec = BinFormat::new()?;
/// let block = codec.encode_block(vec![event.clone()])?;
///
/// let mut reader = std::io::Cursor::new(block);
/// let (decoded, _consumed) = codec.decode_block(&mut reader)?.expect("one block");
/// assert_eq!(decoded, vec![event]);
/// # Ok(())
/// # }
/// ```
pub trait Format: Send {
    /// A short config discriminant (`"bin"`, `"json"`) — what `format = "…"`
    /// in config resolves to.
    fn name(&self) -> &'static str;

    /// Frame + serialize a batch into the exact bytes appended to the log.
    fn encode_block(&mut self, events: Vec<RecordingEvent>) -> Result<Vec<u8>, ProximaError>;

    /// Decode ONE unit from `reader`, returning its events and the byte length
    /// consumed (so the caller advances the cursor). `Ok(None)` at clean EOF.
    /// Reads exactly one unit — never over-reads the underlying source.
    fn decode_block(
        &self,
        reader: &mut dyn BufRead,
    ) -> Result<Option<(Vec<RecordingEvent>, u64)>, ProximaError>;
}
