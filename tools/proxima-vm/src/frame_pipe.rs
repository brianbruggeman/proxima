//! In-process frame channel exposed purely as pipes: [`FrameSink`] is the
//! `Vec<u8> -> ()` sink end and [`FrameSource`] the `() -> Vec<u8>` source end
//! of one bounded lock-free queue. This is the only place a queue appears;
//! the virtio-net device model and whatever plays the network peer see just
//! `Pipe<In = Vec<u8>, Out = ()>` and `Pipe<In = (), Out = Vec<u8>>`, so a
//! proxy hop to another host is the same wiring with a different pipe in
//! place of these two.
//!
//! The types exist only because `Pipe` is foreign to `ArrayQueue`: the orphan
//! rule needs a local type to carry each impl.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam_queue::ArrayQueue;
use proxima_primitives::pipe::Pipe;

/// Why a frame pipe did not move a frame this call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    /// Source: nothing ready right now; try again.
    Empty,
    /// Sink: the queue is at capacity; the frame was not taken.
    Full,
    /// Source: the sink end closed and everything it sent was drained.
    Closed,
}

#[derive(Debug)]
struct Channel {
    queue: ArrayQueue<Vec<u8>>,
    closed: AtomicBool,
}

#[derive(Debug, Clone)]
pub struct FrameSink {
    channel: Arc<Channel>,
}

#[derive(Debug, Clone)]
pub struct FrameSource {
    channel: Arc<Channel>,
}

/// A connected sink and source over one queue holding at most `capacity` frames.
#[must_use]
pub fn frame_channel(capacity: usize) -> (FrameSink, FrameSource) {
    let channel = Arc::new(Channel {
        queue: ArrayQueue::new(capacity),
        closed: AtomicBool::new(false),
    });
    (
        FrameSink {
            channel: channel.clone(),
        },
        FrameSource { channel },
    )
}

impl FrameSink {
    /// Mark the stream finished: the source reports [`FrameError::Closed`]
    /// once it has drained what was already sent.
    pub fn close(&self) {
        self.channel.closed.store(true, Ordering::Release);
    }
}

impl Pipe for FrameSink {
    type In = Vec<u8>;
    type Out = ();
    type Err = FrameError;

    fn call(&self, frame: Vec<u8>) -> impl Future<Output = Result<(), FrameError>> {
        async move {
            self.channel
                .queue
                .push(frame)
                .map_err(|_| FrameError::Full)
        }
    }
}

impl Pipe for FrameSource {
    type In = ();
    type Out = Vec<u8>;
    type Err = FrameError;

    fn call(&self, _input: ()) -> impl Future<Output = Result<Vec<u8>, FrameError>> {
        async move {
            let closed = self.channel.closed.load(Ordering::Acquire);
            match self.channel.queue.pop() {
                Some(frame) => Ok(frame),
                None if closed => Err(FrameError::Closed),
                None => Err(FrameError::Empty),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use futures::FutureExt;

    #[test]
    fn frames_arrive_in_order_then_empty_then_closed() {
        let (sink, source) = frame_channel(4);

        sink.call(vec![1]).now_or_never().unwrap().unwrap();
        sink.call(vec![2]).now_or_never().unwrap().unwrap();

        assert_eq!(source.call(()).now_or_never().unwrap(), Ok(vec![1]));
        assert_eq!(source.call(()).now_or_never().unwrap(), Ok(vec![2]));
        assert_eq!(source.call(()).now_or_never().unwrap(), Err(FrameError::Empty));
        sink.close();
        assert_eq!(source.call(()).now_or_never().unwrap(), Err(FrameError::Closed));
    }

    #[test]
    fn closed_source_still_drains_frames_sent_before_the_close() {
        let (sink, source) = frame_channel(4);
        sink.call(vec![9]).now_or_never().unwrap().unwrap();
        sink.close();

        assert_eq!(source.call(()).now_or_never().unwrap(), Ok(vec![9]));
        assert_eq!(source.call(()).now_or_never().unwrap(), Err(FrameError::Closed));
    }

    #[test]
    fn full_sink_refuses_the_frame() {
        let (sink, _source) = frame_channel(1);
        sink.call(vec![1]).now_or_never().unwrap().unwrap();

        assert_eq!(sink.call(vec![2]).now_or_never().unwrap(), Err(FrameError::Full));
    }
}
