//! Batched terminal writes: accumulate many encoded items into one buffer, then
//! flush the whole accumulation to a sink in a SINGLE write. The discipline that
//! turns N per-item writes into one — the expensive terminal op (a syscall, a
//! DMA submit) is paid once per batch instead of per item, so a producer is
//! never stalled per-item on a slow terminal.
//!
//! [`BatchSink`] is the no_std-clean terminal abstraction: it does NOT name
//! `std::io::Write` (which is std-only), so embedded / UART / DPDK / NVMe
//! terminals can implement it. A `std::io::Write` adapter lives in the consumer
//! crate (e.g. proxima-telemetry wraps a writer as a `BatchSink`) because the
//! orphan rule forbids a blanket `impl BatchSink for W: io::Write` here.
//!
//! [`BatchBuffer`] accumulates encoded bytes — `push_str` for formatted text,
//! `push_bytes` for binary, or `core::fmt::Write` for `write!` — and
//! [`flush_to`](BatchBuffer::flush_to) hands the whole accumulation to a
//! [`BatchSink`] in one call, then clears for reuse.
//!
//! This is the primitive proxima-telemetry's terminal sink composes: per record
//! it appends a formatted line; per drain batch it flushes once.

use alloc::vec::Vec;

/// A terminal that accepts an entire batch of bytes in one operation.
///
/// no_std-clean by construction — no `std::io::Write`. The consumer adapts a
/// concrete writer (a file, a socket, an egress ring) to this trait; the
/// batching machinery above it (`BatchBuffer`) never names the terminal's kind.
pub trait BatchSink {
    /// Terminal-specific failure (an `io::Error`, a device error, ...).
    type Error;

    /// Write the whole batch in one terminal operation.
    fn write_batch(&mut self, bytes: &[u8]) -> Result<(), Self::Error>;
}

/// Accumulates encoded bytes and flushes them to a [`BatchSink`] in one write.
///
/// Append with [`push_str`](Self::push_str) / [`push_bytes`](Self::push_bytes),
/// or via `write!` (it implements [`core::fmt::Write`]); then
/// [`flush_to`](Self::flush_to) writes the accumulation in a single
/// [`BatchSink::write_batch`] and clears for reuse.
#[derive(Debug, Default)]
pub struct BatchBuffer {
    bytes: Vec<u8>,
}

impl BatchBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Preallocate `capacity` bytes — size it to a batch's typical total so the
    /// accumulation does not grow mid-batch.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
        }
    }

    /// Append a UTF-8 string — the common case (a formatted log/metric line).
    pub fn push_str(&mut self, text: &str) {
        self.bytes.extend_from_slice(text.as_bytes());
    }

    /// Append raw bytes — for binary wire formats.
    pub fn push_bytes(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Bytes reserved in the backing allocation (>= the `with_capacity` request).
    /// A batch that stays under this never reallocates mid-accumulation.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Drop the accumulation without flushing (retains the backing allocation).
    pub fn clear(&mut self) {
        self.bytes.clear();
    }

    /// Flush the whole accumulation to `sink` in ONE [`BatchSink::write_batch`],
    /// then clear for reuse. No-op (and no terminal write) when empty.
    pub fn flush_to<S: BatchSink>(&mut self, sink: &mut S) -> Result<(), S::Error> {
        if self.bytes.is_empty() {
            return Ok(());
        }
        sink.write_batch(&self.bytes)?;
        self.bytes.clear();
        Ok(())
    }
}

impl core::fmt::Write for BatchBuffer {
    fn write_str(&mut self, text: &str) -> core::fmt::Result {
        self.push_str(text);
        Ok(())
    }
}

/// Adapts any [`std::io::Write`] to a [`BatchSink`], mapping a whole batch to one
/// `write_all`. Lives here (behind `std`) rather than in each consumer: `BatchSink`
/// is defined in this crate, so implementing it for a wrapper is allowed — the
/// orphan rule only blocks a *third* crate from writing this impl. Std consumers
/// (a telemetry terminal sink, a socket sink, ...) wrap their writer in
/// `WriteSink` instead of re-deriving the bridge.
#[cfg(feature = "std")]
pub struct WriteSink<W>(pub W);

#[cfg(feature = "std")]
impl<W: std::io::Write> BatchSink for WriteSink<W> {
    type Error = std::io::Error;
    fn write_batch(&mut self, bytes: &[u8]) -> Result<(), std::io::Error> {
        self.0.write_all(bytes)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::convert::Infallible;
    use core::fmt::Write as _;

    // A sink that records every write_batch call so a test can prove the batch
    // left the buffer in exactly one terminal write — the whole point.
    #[derive(Default)]
    struct RecordingSink {
        writes: usize,
        bytes: Vec<u8>,
    }

    impl BatchSink for RecordingSink {
        type Error = Infallible;
        fn write_batch(&mut self, bytes: &[u8]) -> Result<(), Infallible> {
            self.writes += 1;
            self.bytes.extend_from_slice(bytes);
            Ok(())
        }
    }

    // The batching contract: many appends, ONE write_batch carrying the
    // concatenation, buffer cleared after.
    #[test]
    fn many_appends_flush_in_one_write() {
        let mut buffer = BatchBuffer::new();
        buffer.push_str("alpha\n");
        buffer.push_str("beta\n");
        buffer.push_str("gamma\n");

        let mut sink = RecordingSink::default();
        buffer.flush_to(&mut sink).unwrap();

        assert_eq!(sink.writes, 1, "three appends flush in a single write");
        assert_eq!(&sink.bytes, b"alpha\nbeta\ngamma\n");
        assert!(buffer.is_empty(), "buffer cleared after flush");
    }

    // Empty flush is a no-op — no spurious zero-length terminal write.
    #[test]
    fn empty_flush_does_not_write() {
        let mut buffer = BatchBuffer::new();
        let mut sink = RecordingSink::default();
        buffer.flush_to(&mut sink).unwrap();
        assert_eq!(sink.writes, 0, "no write for an empty batch");
    }

    // Reuse: a second batch after a flush starts clean and flushes once more.
    #[test]
    fn buffer_reuses_after_flush() {
        let mut buffer = BatchBuffer::with_capacity(64);
        let mut sink = RecordingSink::default();

        buffer.push_str("first\n");
        buffer.flush_to(&mut sink).unwrap();
        buffer.push_str("second\n");
        buffer.flush_to(&mut sink).unwrap();

        assert_eq!(sink.writes, 2, "one write per batch");
        assert_eq!(&sink.bytes, b"first\nsecond\n");
    }

    // push_bytes carries arbitrary (non-UTF-8) bytes for binary terminals.
    #[test]
    fn push_bytes_carries_raw_bytes() {
        let mut buffer = BatchBuffer::new();
        buffer.push_bytes(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let mut sink = RecordingSink::default();
        buffer.flush_to(&mut sink).unwrap();
        assert_eq!(&sink.bytes, &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    // with_capacity reserves up front so a batch under that size never
    // reallocates mid-accumulation (the sized-floor preallocation).
    #[test]
    fn with_capacity_preallocates() {
        let buffer = BatchBuffer::with_capacity(8192);
        assert!(buffer.capacity() >= 8192, "reserves the requested bytes");
        assert!(buffer.is_empty(), "reserved but holds nothing yet");
    }

    // the std WriteSink adapter turns any io::Write into a BatchSink — many
    // appends land in one write_all on the underlying writer.
    #[cfg(feature = "std")]
    #[test]
    fn write_sink_adapts_io_write() {
        let mut out: Vec<u8> = Vec::new();
        let mut buffer = BatchBuffer::new();
        buffer.push_str("alpha\n");
        buffer.push_str("beta\n");
        buffer.flush_to(&mut WriteSink(&mut out)).unwrap();
        assert_eq!(&out, b"alpha\nbeta\n");
    }

    // core::fmt::Write lets a caller format straight into the batch.
    #[test]
    fn fmt_write_appends_formatted_output() {
        let mut buffer = BatchBuffer::new();
        let seq = 12_345;
        let peer = "10.0.0.1:443";
        writeln!(buffer, "seq={seq} peer={peer}").unwrap();
        let mut sink = RecordingSink::default();
        buffer.flush_to(&mut sink).unwrap();
        assert_eq!(&sink.bytes, b"seq=12345 peer=10.0.0.1:443\n");
    }
}
