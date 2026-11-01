// a bench's setup has no caller to return an error to; a failed setup must abort the run.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::hint::black_box;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use crossbeam_queue::ArrayQueue;
use futures::FutureExt;
use proxima_primitives::pipe::Pipe;
use proxima_vm::frame_pipe::{FrameError, FrameSink, FrameSource, frame_channel, frame_channel_waking};

const SIZES: [usize; 3] = [54, 402, 1514];
const CAPACITY: usize = 256;
const NEUTRAL: &str = "neutral";
const FAVORS_INCUMBENT: &str = "design-favors-incumbent";

fn same_thread_send_recv(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("frame_pipe/same_thread");
    group.throughput(Throughput::Elements(1));
    for size in SIZES {
        let (sink, source) = frame_channel(CAPACITY);
        let mut frame = vec![0xA5_u8; size];
        let id = BenchmarkId::new(format!("{NEUTRAL}/proxima_send+recv"), size);
        group.bench_function(id, |bencher| {
            bencher.iter(|| {
                sink.call(std::mem::take(&mut frame))
                    .now_or_never()
                    .expect("sink ready")
                    .expect("queue has room");
                frame = source
                    .call(())
                    .now_or_never()
                    .expect("frame queued")
                    .expect("source open");
                black_box(&frame);
            });
        });
    }
    group.finish();
}

fn proxima_echo() -> (FrameSink, FrameSource, JoinHandle<()>) {
    let (handoff, receiver) = mpsc::channel::<FrameSink>();
    let (back_sink, back_source) = frame_channel_waking(CAPACITY, thread::current());
    let echo = thread::spawn(move || {
        let (forward_sink, forward_source) = frame_channel_waking(CAPACITY, thread::current());
        handoff
            .send(forward_sink)
            .expect("handoff forward sink");
        loop {
            match forward_source.call(()).now_or_never() {
                Some(Ok(frame)) => {
                    back_sink
                        .call(frame)
                        .now_or_never()
                        .expect("back sink ready")
                        .expect("back queue has room");
                }
                None => thread::park(),
                Some(Err(FrameError::Closed)) => return,
                Some(Err(FrameError::Full)) => unreachable!("a source is never full"),
            }
        }
    });
    let forward_sink = receiver.recv().expect("forward sink handed over");
    (forward_sink, back_source, echo)
}

fn proxima_cross_thread_ping(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("frame_pipe/cross_thread_ping");
    group.throughput(Throughput::Elements(1));
    for size in SIZES {
        let (forward, back, echo) = proxima_echo();
        let mut frame = vec![0xA5_u8; size];
        let id = BenchmarkId::new(format!("{NEUTRAL}/proxima_frame_channel_waking"), size);
        group.bench_function(id, |bencher| {
            bencher.iter(|| {
                forward
                    .call(std::mem::take(&mut frame))
                    .now_or_never()
                    .expect("sink ready")
                    .expect("queue has room");
                frame = loop {
                    match back.call(()).now_or_never() {
                        Some(Ok(returned)) => break returned,
                        None => thread::park(),
                        Some(Err(error)) => panic!("back source failed: {error:?}"),
                    }
                };
                black_box(&frame);
            });
        });
        forward.close();
        echo.join().expect("echo thread exits on close");
    }
    group.finish();
}


fn crossbeam_cross_thread_ping(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("frame_pipe/cross_thread_ping");
    group.throughput(Throughput::Elements(1));
    for size in SIZES {
        let forward = Arc::new(ArrayQueue::<Vec<u8>>::new(CAPACITY));
        let back = Arc::new(ArrayQueue::<Vec<u8>>::new(CAPACITY));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let main_thread = thread::current();
        let echo = {
            let (forward, back, stop) = (forward.clone(), back.clone(), stop.clone());
            thread::spawn(move || {
                loop {
                    match forward.pop() {
                        Some(frame) => {
                            back.push(frame).expect("back queue has room");
                            main_thread.unpark();
                        }
                        None if stop.load(std::sync::atomic::Ordering::Acquire) => return,
                        None => thread::park(),
                    }
                }
            })
        };
        let mut frame = vec![0xA5_u8; size];
        let id = BenchmarkId::new(format!("{FAVORS_INCUMBENT}/crossbeam_arrayqueue+park_unpark"), size);
        group.bench_function(id, |bencher| {
            bencher.iter(|| {
                forward.push(std::mem::take(&mut frame)).expect("queue has room");
                echo.thread().unpark();
                frame = loop {
                    match back.pop() {
                        Some(returned) => break returned,
                        None => thread::park(),
                    }
                };
                black_box(&frame);
            });
        });
        stop.store(true, std::sync::atomic::Ordering::Release);
        echo.thread().unpark();
        echo.join().expect("echo thread exits on stop");
    }
    group.finish();
}

fn mpsc_cross_thread_ping(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("frame_pipe/cross_thread_ping");
    group.throughput(Throughput::Elements(1));
    for size in SIZES {
        let (forward_sender, forward_receiver) = mpsc::sync_channel::<Vec<u8>>(CAPACITY);
        let (back_sender, back_receiver) = mpsc::sync_channel::<Vec<u8>>(CAPACITY);
        let echo = thread::spawn(move || {
            while let Ok(frame) = forward_receiver.recv() {
                if back_sender.send(frame).is_err() {
                    return;
                }
            }
        });
        let mut frame = vec![0xA5_u8; size];
        let id = BenchmarkId::new(format!("{FAVORS_INCUMBENT}/std_mpsc_sync_channel"), size);
        group.bench_function(id, |bencher| {
            bencher.iter(|| {
                forward_sender.send(std::mem::take(&mut frame)).expect("echo alive");
                frame = back_receiver.recv().expect("echo replies");
                black_box(&frame);
            });
        });
        drop(forward_sender);
        echo.join().expect("echo thread exits on disconnect");
    }
    group.finish();
}

criterion_group!(
    benches,
    same_thread_send_recv,
    proxima_cross_thread_ping,
    crossbeam_cross_thread_ping,
    mpsc_cross_thread_ping
);
criterion_main!(benches);
