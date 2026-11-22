//! Publishes a reaped child's end into the response stream's trailers.
//!
//! Composes [`ResponseStream::with_trailers_slot`] with a
//! [`futures::channel::oneshot`]: the blocking `waitpid` runs on its own thread
//! (never on the executor) and the body stream awaits the result before it
//! ends, as HTTP trailers complete a response.
use std::thread;

use futures::StreamExt;
use futures::channel::oneshot;
use futures::stream;
use proxima_primitives::pipe::{ChunkStream, HeaderList, ResponseStream};
use proxima_primitives::sync::blocking::Mutex;
use std::sync::Arc;

/// Decode a raw unix wait status: a normal exit becomes `exit-status`, a
/// signal death becomes `exit-signal`, never both. A stopped or unknown status
/// yields no trailer.
#[must_use]
pub(crate) fn exit_trailers(status: libc::c_int) -> HeaderList {
    let mut trailers = HeaderList::new();
    if libc::WIFEXITED(status) {
        trailers.insert("exit-status", libc::WEXITSTATUS(status).to_string());
    } else if libc::WIFSIGNALED(status) {
        trailers.insert("exit-signal", libc::WTERMSIG(status).to_string());
    }
    trailers
}

/// Blocking `waitpid` for one child; `None` when the kernel refuses.
#[must_use]
pub(crate) fn wait_status(pid: libc::pid_t) -> Option<libc::c_int> {
    let mut status: libc::c_int = 0;
    // SAFETY: waitpid takes our own child's pid and a pointer to a stack int.
    let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
    (waited == pid).then_some(status)
}

/// Wrap `body` so that, once it drains, `reap` runs on a detached thread and
/// the stream awaits its raw wait status, publishes [`exit_trailers`] into the
/// stream's trailers slot, and only then ends.
#[must_use]
pub(crate) fn body_with_exit_trailers<Reap>(body: ChunkStream, reap: Reap) -> ResponseStream
where
    Reap: FnOnce() -> Option<libc::c_int> + Send + 'static,
{
    let slot = Arc::new(Mutex::new(None));
    let publish = Arc::clone(&slot);
    let stream = stream::unfold(
        (body, Some(reap), publish),
        |(mut body, mut reap, publish)| async move {
            match body.next().await {
                Some(item) => Some((item, (body, reap, publish))),
                None => {
                    if let Some(reap) = reap.take() {
                        let (sender, receiver) = oneshot::channel();
                        thread::spawn(move || {
                            if let Some(status) = reap() {
                                let _ = sender.send(status);
                            }
                        });
                        if let Ok(status) = receiver.await {
                            *publish.lock() = Some(exit_trailers(status));
                        }
                    }
                    None
                }
            }
        },
    );
    ResponseStream::new(stream).with_trailers_slot(slot)
}

#[cfg(test)]
pub(crate) mod test_support {
    use bytes::Bytes;
    use futures::StreamExt;
    use futures::executor::block_on;
    use proxima_primitives::pipe::{HeaderList, Request, Response};

    #[must_use]
    pub(crate) fn empty_request() -> Request<Bytes> {
        Request::builder()
            .method("POST")
            .path("/")
            .build()
            .expect("request builds with default empty body")
    }

    /// Drain the body to its end, then read the trailers the stream published.
    #[must_use]
    pub(crate) fn drain_then_trailers(response: Response<Bytes>) -> HeaderList {
        let mut stream = response.stream.expect("child response is streamed");
        block_on(async {
            while let Some(chunk) = stream.chunk_stream_mut().next().await {
                chunk.expect("body chunk ok");
            }
        });
        stream
            .trailers()
            .expect("trailers published before the body ended")
    }

    pub(crate) fn assert_exit(trailers: &HeaderList, code: &str) {
        assert_eq!(trailers.get_str("exit-status"), Some(code));
        assert!(!trailers.contains_key("exit-signal"));
    }

    pub(crate) fn assert_signal(trailers: &HeaderList, signal: &str) {
        assert_eq!(trailers.get_str("exit-signal"), Some(signal));
        assert!(!trailers.contains_key("exit-status"));
    }
}
