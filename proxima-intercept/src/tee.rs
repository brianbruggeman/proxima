//! Token-delta tee: best-effort broadcast of raw chunk bytes to subscribers.
//!
//! Gated behind the `delta-tee` feature. When the feature is off, this
//! module compiles to nothing and the call sites are removed by cfg.

#[cfg(feature = "delta-tee")]
use bytes::Bytes;
#[cfg(feature = "delta-tee")]
use proxima_primitives::sync::broadcast::Sender;

/// Forward a chunk to every subscribed receiver via a best-effort broadcast.
///
/// "Best-effort" means: if no receivers are listening, or if a receiver
/// has fallen behind and the buffer overflowed, the response is not
/// failed — the error is silently discarded. The main response path is
/// never interrupted by tee failures.
#[cfg(feature = "delta-tee")]
pub fn tee_chunk(tx: &Option<Sender<Bytes>>, chunk: &Bytes) {
    if let Some(sender) = tx {
        let _ = sender.send(chunk.clone());
    }
}

#[cfg(all(test, feature = "delta-tee"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use proxima_primitives::sync::broadcast;

    #[test]
    fn tee_chunk_with_active_receiver_delivers_exact_bytes_in_order() {
        let (tx, mut rx) = broadcast::channel::<Bytes>(8);
        let sender = Some(tx);

        let chunks: Vec<Bytes> = vec![
            Bytes::from(r#"data: {"type":"response.output_text.delta","delta":"Hello"}"#),
            Bytes::from(r#"data: {"type":"response.output_text.delta","delta":" world"}"#),
            Bytes::from(r#"data: {"type":"response.completed"}"#),
        ];

        for chunk in &chunks {
            tee_chunk(&sender, chunk);
        }

        block_on(async {
            for expected in &chunks {
                let received = rx.recv().await.expect("chunk delivered");
                assert_eq!(received, *expected);
            }
        });
    }

    #[test]
    fn tee_chunk_with_no_receivers_does_not_panic() {
        let (tx, rx) = broadcast::channel::<Bytes>(8);
        drop(rx);
        let sender = Some(tx);
        let chunk = Bytes::from(r#"data: {"type":"response.output_text.delta","delta":"hi"}"#);
        tee_chunk(&sender, &chunk);
    }

    #[test]
    fn tee_chunk_with_none_sender_is_noop() {
        let sender: Option<Sender<Bytes>> = None;
        let chunk = Bytes::from(r#"data: {"type":"response.output_text.delta","delta":"hi"}"#);
        tee_chunk(&sender, &chunk);
    }

    #[test]
    fn tee_chunk_does_not_copy_buffer_contents() {
        let (tx, mut rx) = broadcast::channel::<Bytes>(4);
        let sender = Some(tx);
        let original = Bytes::from_static(b"delta bytes");
        tee_chunk(&sender, &original);
        let received = block_on(async { rx.recv().await.expect("recv") });
        assert_eq!(
            received.as_ptr(),
            original.as_ptr(),
            "refcount clone, not a copy"
        );
    }

    #[test]
    fn tee_chunk_lagging_receiver_does_not_block_caller() {
        let capacity = 2;
        let (tx, _rx) = broadcast::channel::<Bytes>(capacity);
        let sender = Some(tx.clone());
        let late_rx = tx.subscribe();
        drop(late_rx);
        for index in 0..10u8 {
            let chunk = Bytes::copy_from_slice(&[index]);
            tee_chunk(&sender, &chunk);
        }
    }
}
