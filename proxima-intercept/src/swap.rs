//! The generic swap seam for the intercept proxy. The [`Turn`] + [`SwapSurface`]
//! definitions now live canonically in `proxima-pipe` (sans-IO, no_std+alloc) and
//! are re-exported here so existing `proxima_intercept::{Turn, SwapSurface}` paths
//! keep resolving. This module also holds the IO half: the generic synth pump that
//! drives any [`StreamFramer`](proxima_primitives::pipe::StreamFramer) onto a writer.

pub use proxima_primitives::pipe::{StreamFramer, SwapSurface, Turn};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use tokio::io::{AsyncWrite, AsyncWriteExt};

/// Drive a synthesized SSE answer onto a writer from a **live stream** of prose
/// deltas — the streaming swap path. Writes the caller-supplied head + opening,
/// re-frames each delta as it arrives (flushed per chunk), then closing — or a
/// terminal error frame on a mid-stream `Err` (no closing after a failure). Vocab-
/// free: the `head` bytes and the framer's frames carry all vendor knowledge.
pub async fn pump_synthesized_sse<Writer, Framer, Deltas, DeltaError>(
    writer: &mut Writer,
    head: &[u8],
    framer: &mut Framer,
    deltas: Deltas,
) -> std::io::Result<()>
where
    Writer: AsyncWrite + Unpin,
    Framer: proxima_primitives::pipe::StreamFramer,
    Deltas: Stream<Item = Result<Bytes, DeltaError>>,
    DeltaError: core::fmt::Display,
{
    writer.write_all(head).await?;
    writer.write_all(&framer.opening()).await?;
    writer.flush().await?;

    let mut deltas = core::pin::pin!(deltas);
    while let Some(item) = deltas.next().await {
        match item {
            Ok(delta) => {
                let text = String::from_utf8_lossy(&delta);
                writer.write_all(&framer.delta(&text)).await?;
                writer.flush().await?;
            }
            Err(err) => {
                writer.write_all(&framer.error(&err.to_string())).await?;
                writer.flush().await?;
                return Ok(());
            }
        }
    }

    writer.write_all(&framer.closing()).await?;
    writer.flush().await?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Deterministic in-test framer with literal frames, mirroring the control-flow
    /// shapes a real vocab framer emits (opening preamble, per-delta frame with the
    /// delta text, a closing frame on success, a terminal error frame on failure).
    #[derive(Default)]
    struct TestFramer {
        accumulated: String,
    }

    impl StreamFramer for TestFramer {
        fn opening(&mut self) -> Vec<u8> {
            b"event: opening\n".to_vec()
        }

        fn delta(&mut self, text: &str) -> Vec<u8> {
            self.accumulated.push_str(text);
            format!("event: delta\ndelta: {text}\n").into_bytes()
        }

        fn closing(&mut self) -> Vec<u8> {
            format!("event: closing\ntext: {}\n", self.accumulated).into_bytes()
        }

        fn error(&mut self, message: &str) -> Vec<u8> {
            format!("event: failed\nerror: {message}\n").into_bytes()
        }
    }

    #[proxima::test]
    async fn pump_writes_head_opening_deltas_closing_from_stream() {
        let mut sink: Vec<u8> = Vec::new();
        let mut framer = TestFramer::default();
        let head = b"HTTP/1.1 200 OK\r\n\r\n";
        let deltas = futures::stream::iter(vec![
            Ok::<_, std::io::Error>(Bytes::from("Hello ")),
            Ok(Bytes::from("world")),
        ]);
        pump_synthesized_sse(&mut sink, head, &mut framer, deltas)
            .await
            .expect("pump");
        let out = String::from_utf8(sink).unwrap();
        assert!(out.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(out.contains("event: opening\n"));
        assert_eq!(out.matches("event: delta\n").count(), 2);
        assert!(out.contains("delta: Hello \n"));
        assert!(out.contains("text: Hello world\n"));
        assert!(out.contains("event: closing\n"));
    }

    #[proxima::test]
    async fn pump_mid_stream_error_emits_failed_not_closing() {
        let mut sink: Vec<u8> = Vec::new();
        let mut framer = TestFramer::default();
        let head = b"HTTP/1.1 200 OK\r\n\r\n";
        let deltas = futures::stream::iter(vec![
            Ok(Bytes::from("partial ")),
            Err(std::io::Error::other("upstream reset")),
        ]);
        pump_synthesized_sse(&mut sink, head, &mut framer, deltas)
            .await
            .expect("pump");
        let out = String::from_utf8(sink).unwrap();
        assert!(out.contains("delta: partial \n"));
        assert!(out.contains("event: failed\n"));
        assert!(!out.contains("event: closing"));
    }
}
