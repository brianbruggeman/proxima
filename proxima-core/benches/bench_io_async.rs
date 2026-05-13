use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::io::AsyncRead as FuturesAsyncRead;
use proxima_core::io::AsyncRead as ProximaAsyncRead;

// a trait cannot be "faster" than another trait — this bench proves ZERO
// OVERHEAD against the std-only incumbent (futures::io::AsyncRead), not a
// win. arm A (proxima) and arm B (futures-io) run the identical poll loop
// over the identical bytes; parity is the expected, honest outcome.
const SIZES: &[usize] = &[256, 16_384];
const CHUNK: usize = 64;

struct ProximaSliceReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ProximaSliceReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl ProximaAsyncRead for ProximaSliceReader<'_> {
    type Error = core::convert::Infallible;

    fn poll_read(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<Result<usize, Self::Error>> {
        let this = self.get_mut();
        let remaining = &this.data[this.pos..];
        let count = remaining.len().min(buf.len());
        buf[..count].copy_from_slice(&remaining[..count]);
        this.pos += count;
        Poll::Ready(Ok(count))
    }
}

struct FuturesSliceReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> FuturesSliceReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl FuturesAsyncRead for FuturesSliceReader<'_> {
    fn poll_read(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        let this = self.get_mut();
        let remaining = &this.data[this.pos..];
        let count = remaining.len().min(buf.len());
        buf[..count].copy_from_slice(&remaining[..count]);
        this.pos += count;
        Poll::Ready(Ok(count))
    }
}

fn drain_proxima(data: &[u8], context: &mut Context<'_>) -> u64 {
    let mut reader = ProximaSliceReader::new(data);
    let mut chunk = [0u8; CHUNK];
    let mut total = 0u64;

    loop {
        match Pin::new(&mut reader).poll_read(context, &mut chunk) {
            Poll::Ready(Ok(0)) => break,
            Poll::Ready(Ok(count)) => {
                total += chunk[..count]
                    .iter()
                    .map(|&byte| u64::from(byte))
                    .sum::<u64>();
            }
            Poll::Ready(Err(never)) => match never {},
            Poll::Pending => unreachable!("ProximaSliceReader never pends"),
        }
    }

    total
}

fn drain_futures(data: &[u8], context: &mut Context<'_>) -> u64 {
    let mut reader = FuturesSliceReader::new(data);
    let mut chunk = [0u8; CHUNK];
    let mut total = 0u64;

    loop {
        match Pin::new(&mut reader).poll_read(context, &mut chunk) {
            Poll::Ready(Ok(0)) => break,
            Poll::Ready(Ok(count)) => {
                total += chunk[..count]
                    .iter()
                    .map(|&byte| u64::from(byte))
                    .sum::<u64>();
            }
            Poll::Ready(Err(error)) => unreachable!("FuturesSliceReader never errors: {error}"),
            Poll::Pending => unreachable!("FuturesSliceReader never pends"),
        }
    }

    total
}

fn proxima_passthrough(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("io_parity");
    let mut context = Context::from_waker(Waker::noop());

    for size in SIZES {
        let data = vec![0xABu8; *size];
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("proxima_neutral", size),
            &data,
            |bench, data| {
                bench.iter(|| std::hint::black_box(drain_proxima(data, &mut context)));
            },
        );
    }

    group.finish();
}

fn futures_io_passthrough(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("io_parity");
    let mut context = Context::from_waker(Waker::noop());

    for size in SIZES {
        let data = vec![0xABu8; *size];
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("futures_io_incumbent", size),
            &data,
            |bench, data| {
                bench.iter(|| std::hint::black_box(drain_futures(data, &mut context)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, proxima_passthrough, futures_io_passthrough);
criterion_main!(benches);
