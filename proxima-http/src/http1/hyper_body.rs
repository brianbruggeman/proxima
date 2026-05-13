use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::stream::StreamExt;
use http_body::{Body as HttpBody, Frame, SizeHint};
use pin_project_lite::pin_project;

use proxima_core::ProximaError;
use proxima_primitives::pipe::body::ChunkStream;

pin_project! {
    pub struct StreamingHyperBody {
        inner: ChunkStream,
        finished: bool,
    }
}

impl StreamingHyperBody {
    #[must_use]
    pub fn new(stream: ChunkStream) -> Self {
        Self {
            inner: stream,
            finished: false,
        }
    }
}

impl HttpBody for StreamingHyperBody {
    type Data = Bytes;
    type Error = ProximaError;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.project();
        if *this.finished {
            return Poll::Ready(None);
        }
        match this.inner.poll_next_unpin(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                *this.finished = true;
                Poll::Ready(None)
            }
            Poll::Ready(Some(Ok(bytes))) => Poll::Ready(Some(Ok(Frame::data(bytes)))),
            Poll::Ready(Some(Err(error))) => {
                *this.finished = true;
                Poll::Ready(Some(Err(error)))
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        self.finished
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;

    #[proxima::test]
    async fn streaming_hyper_body_yields_chunks_in_order() {
        let stream: ChunkStream = Box::pin(futures::stream::iter([
            Ok(Bytes::from_static(b"hel")),
            Ok(Bytes::from_static(b"lo")),
        ]));
        let collected = StreamingHyperBody::new(stream)
            .collect()
            .await
            .expect("collect")
            .to_bytes();
        assert_eq!(&collected[..], b"hello");
    }

    #[proxima::test]
    async fn streaming_hyper_body_propagates_error() {
        let stream: ChunkStream = Box::pin(futures::stream::iter([
            Ok(Bytes::from_static(b"a")),
            Err(ProximaError::Body("boom".into())),
        ]));
        let outcome = StreamingHyperBody::new(stream).collect().await;
        assert!(outcome.is_err());
    }
}
