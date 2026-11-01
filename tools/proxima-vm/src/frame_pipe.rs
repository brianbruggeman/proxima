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

use std::future::poll_fn;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::Poll;
use std::thread::Thread;

use crossbeam_queue::ArrayQueue;
use futures::task::AtomicWaker;
use proxima_primitives::pipe::Pipe;

/// Why a frame pipe did not move a frame this call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    /// Sink: the queue is at capacity; the frame was not taken.
    Full,
    /// Source: the sink end closed and everything it sent was drained.
    Closed,
}

#[derive(Debug)]
struct Channel {
    queue: ArrayQueue<Vec<u8>>,
    closed: AtomicBool,
    consumer: AtomicWaker,
}

#[derive(Debug, Clone)]
pub struct FrameSink {
    channel: Arc<Channel>,
    wake: Option<Thread>,
}

#[derive(Debug, Clone)]
pub struct FrameSource {
    channel: Arc<Channel>,
}

/// A connected sink and source over one queue holding at most `capacity` frames.
#[must_use]
pub fn frame_channel(capacity: usize) -> (FrameSink, FrameSource) {
    connected(capacity, None)
}

/// [`frame_channel`] whose sink unparks `consumer` after every frame it
/// accepts or that is closed, so a consumer thread can park between frames
/// instead of polling.
#[must_use]
pub fn frame_channel_waking(capacity: usize, consumer: Thread) -> (FrameSink, FrameSource) {
    connected(capacity, Some(consumer))
}

fn connected(capacity: usize, wake: Option<Thread>) -> (FrameSink, FrameSource) {
    let channel = Arc::new(Channel {
        queue: ArrayQueue::new(capacity),
        closed: AtomicBool::new(false),
        consumer: AtomicWaker::new(),
    });
    (
        FrameSink {
            channel: channel.clone(),
            wake,
        },
        FrameSource { channel },
    )
}

impl FrameSink {
    /// Mark the stream finished: the source reports [`FrameError::Closed`]
    /// once it has drained what was already sent.
    pub fn close(&self) {
        self.channel.closed.store(true, Ordering::Release);
        self.channel.consumer.wake();
        if let Some(consumer) = &self.wake {
            consumer.unpark();
        }
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
                .map_err(|_| FrameError::Full)?;
            self.channel.consumer.wake();
            if let Some(consumer) = &self.wake {
                consumer.unpark();
            }
            Ok(())
        }
    }
}

impl Pipe for FrameSource {
    type In = ();
    type Out = Vec<u8>;
    type Err = FrameError;

    fn call(&self, _input: ()) -> impl Future<Output = Result<Vec<u8>, FrameError>> {
        poll_fn(|context| {
            if let Some(frame) = self.channel.queue.pop() {
                return Poll::Ready(Ok(frame));
            }
            self.channel.consumer.register(context.waker());
            let closed = self.channel.closed.load(Ordering::Acquire);
            match self.channel.queue.pop() {
                Some(frame) => Poll::Ready(Ok(frame)),
                None if closed => Poll::Ready(Err(FrameError::Closed)),
                None => Poll::Pending,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;
    use std::task::Context;
    use std::thread;

    use super::*;
    use futures::FutureExt;
    use futures::task::{ArcWake, waker};

    #[test]
    fn frames_arrive_in_order_then_empty_then_closed() {
        let (sink, source) = frame_channel(4);

        sink.call(vec![1]).now_or_never().unwrap().unwrap();
        sink.call(vec![2]).now_or_never().unwrap().unwrap();

        assert_eq!(source.call(()).now_or_never().unwrap(), Ok(vec![1]));
        assert_eq!(source.call(()).now_or_never().unwrap(), Ok(vec![2]));
        assert!(source.call(()).now_or_never().is_none(), "an empty source is pending, not an error");
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
    fn a_waking_sink_unparks_the_consumer_after_a_push_and_after_a_close() {
        let (ready_sender, ready_receiver) = mpsc::channel();
        let consumer = thread::spawn(move || {
            let (sink, source) = frame_channel_waking(4, thread::current());
            ready_sender.send(sink).unwrap();
            let mut seen = Vec::new();
            loop {
                match source.call(()).now_or_never() {
                    Some(Ok(frame)) => seen.push(frame),
                    None => thread::park(),
                    Some(Err(FrameError::Closed)) => return seen,
                    Some(Err(FrameError::Full)) => unreachable!("a source is never full"),
                }
            }
        });
        let sink = ready_receiver.recv().unwrap();

        sink.call(vec![1]).now_or_never().unwrap().unwrap();
        sink.call(vec![2]).now_or_never().unwrap().unwrap();
        sink.close();

        assert_eq!(consumer.join().unwrap(), vec![vec![1], vec![2]]);
    }

    struct WakeCount(AtomicUsize);

    impl ArcWake for WakeCount {
        fn wake_by_ref(arc_self: &Arc<Self>) {
            arc_self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn a_pending_source_wakes_its_task_on_a_push_and_not_before() {
        let (sink, source) = frame_channel(4);
        let wakes = Arc::new(WakeCount(AtomicUsize::new(0)));
        let task_waker = waker(Arc::clone(&wakes));
        let mut context = Context::from_waker(&task_waker);
        let mut receive = std::pin::pin!(source.call(()));

        assert!(receive.as_mut().poll(&mut context).is_pending());
        assert_eq!(wakes.0.load(Ordering::SeqCst), 0, "a pending poll never wakes itself");
        sink.call(vec![7]).now_or_never().unwrap().unwrap();

        assert_eq!(wakes.0.load(Ordering::SeqCst), 1, "the push woke the parked task");
        assert_eq!(receive.as_mut().poll(&mut context), Poll::Ready(Ok(vec![7])));
    }

    #[test]
    fn full_sink_refuses_the_frame() {
        let (sink, _source) = frame_channel(1);
        sink.call(vec![1]).now_or_never().unwrap().unwrap();

        assert_eq!(sink.call(vec![2]).now_or_never().unwrap(), Err(FrameError::Full));
    }
}
