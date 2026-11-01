// a bench's setup has no caller to return an error to; a failed setup must abort the run.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::FutureExt;
use proxima_primitives::pipe::Pipe;
use proxima_protocols::virtio::mmio::{
    REG_DRIVER_FEATURES, REG_DRIVER_FEATURES_SEL, REG_QUEUE_DESC_HIGH, REG_QUEUE_DESC_LOW,
    REG_QUEUE_DEVICE_HIGH, REG_QUEUE_DEVICE_LOW, REG_QUEUE_DRIVER_HIGH, REG_QUEUE_DRIVER_LOW,
    REG_QUEUE_NUM, REG_QUEUE_READY, REG_QUEUE_SEL, REG_STATUS,
};
use proxima_protocols::virtio::status::{
    STATUS_ACKNOWLEDGE, STATUS_DRIVER, STATUS_DRIVER_OK, STATUS_FEATURES_OK,
};
use proxima_protocols::virtio::{
    DriverQueue, MmioAccess, NET_HDR_LEN, NetHdrFields, USED_ELEM_LEN, write_net_hdr,
};
use proxima_vm::frame_pipe::{FrameSink, FrameSource, frame_channel};
use proxima_vm::virtio_net::{HostNetTransport, NetTransport, RX_QUEUE, TX_QUEUE};

const SIZES: [usize; 3] = [54, 402, 1514];
const QUEUE_SIZE: u16 = 4;
const BATCH: usize = 4;
const OUR_MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
const TX_TABLE: u64 = 0x1000;
const TX_AVAIL: u64 = 0x2000;
const TX_USED: u64 = 0x3000;
const RX_TABLE: u64 = 0x4000;
const RX_AVAIL: u64 = 0x5000;
const RX_USED: u64 = 0x6000;
const BUFFERS: u64 = 0x8000;
const BUFFER_STRIDE: u64 = 0x800;
const MEMORY_SIZE: usize = 0x10000;
const NEUTRAL: &str = "neutral";

fn write(offset: u64, value: u32) -> MmioAccess {
    MmioAccess {
        offset,
        is_write: true,
        value,
    }
}

fn set_up_queue(transport: &mut HostNetTransport, queue: u16, rings: (u64, u64, u64)) {
    let (table, avail, used) = rings;
    let registers = [
        (REG_QUEUE_SEL, u32::from(queue)),
        (REG_QUEUE_NUM, u32::from(QUEUE_SIZE)),
        (REG_QUEUE_DESC_LOW, table as u32),
        (REG_QUEUE_DESC_HIGH, 0),
        (REG_QUEUE_DRIVER_LOW, avail as u32),
        (REG_QUEUE_DRIVER_HIGH, 0),
        (REG_QUEUE_DEVICE_LOW, used as u32),
        (REG_QUEUE_DEVICE_HIGH, 0),
        (REG_QUEUE_READY, 1),
    ];
    for (offset, value) in registers {
        transport.apply(write(offset, value)).expect("queue register write");
    }
}

fn wired_transport() -> (HostNetTransport, FrameSource, FrameSink) {
    let (to_guest, ingress) = frame_channel(8);
    let (egress, from_guest) = frame_channel(8);
    let mut transport = NetTransport::with_pipes(OUR_MAC, ingress, egress);
    let early = [STATUS_ACKNOWLEDGE, STATUS_ACKNOWLEDGE | STATUS_DRIVER];
    for status in early {
        transport.apply(write(REG_STATUS, u32::from(status))).expect("status");
    }
    for (select, word) in [(0, 0), (1, 1)] {
        transport.apply(write(REG_DRIVER_FEATURES_SEL, select)).expect("feature select");
        transport.apply(write(REG_DRIVER_FEATURES, word)).expect("feature word");
    }
    let features_ok = STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK;
    transport.apply(write(REG_STATUS, u32::from(features_ok))).expect("features ok");
    set_up_queue(&mut transport, RX_QUEUE, (RX_TABLE, RX_AVAIL, RX_USED));
    set_up_queue(&mut transport, TX_QUEUE, (TX_TABLE, TX_AVAIL, TX_USED));
    let driver_ok = features_ok | STATUS_DRIVER_OK;
    transport.apply(write(REG_STATUS, u32::from(driver_ok))).expect("driver ok");
    (transport, from_guest, to_guest)
}

struct Guest {
    memory: Vec<u8>,
    tx: DriverQueue<4>,
    rx: DriverQueue<4>,
}

impl Guest {
    fn new() -> Self {
        let mut memory = vec![0_u8; MEMORY_SIZE];
        let tx = DriverQueue::<4>::new(&mut memory[TX_TABLE as usize..]).expect("tx table");
        let rx = DriverQueue::<4>::new(&mut memory[RX_TABLE as usize..]).expect("rx table");
        Self { memory, tx, rx }
    }

    fn buffer_address(slot: usize) -> u64 {
        BUFFERS + slot as u64 * BUFFER_STRIDE
    }

    fn transmit(&mut self, slot: usize, frame: &[u8]) {
        let address = Self::buffer_address(slot);
        let mut header = [0_u8; NET_HDR_LEN];
        write_net_hdr(&mut header, NetHdrFields { num_buffers: 1, ..NetHdrFields::default() })
            .expect("net hdr");
        let start = address as usize;
        self.memory[start..start + NET_HDR_LEN].copy_from_slice(&header);
        self.memory[start + NET_HDR_LEN..start + NET_HDR_LEN + frame.len()].copy_from_slice(frame);
        let length = (NET_HDR_LEN + frame.len()) as u32;
        let (before, after) = self.memory.split_at_mut(TX_AVAIL as usize);
        self.tx
            .post(&mut before[TX_TABLE as usize..], after, &[(address, length)], &[])
            .expect("tx post");
    }

    fn post_rx_buffer(&mut self, slot: usize) {
        let address = Self::buffer_address(slot);
        let (before, after) = self.memory.split_at_mut(RX_AVAIL as usize);
        self.rx
            .post(&mut before[RX_TABLE as usize..], after, &[], &[(address, BUFFER_STRIDE as u32)])
            .expect("rx post");
    }

    fn reap_tx(&mut self) {
        let used = self.memory[TX_USED as usize..TX_USED as usize + 4 + 4 * USED_ELEM_LEN].to_vec();
        self.tx
            .reap(&mut self.memory[TX_TABLE as usize..], &used)
            .expect("tx reap")
            .expect("tx completion published");
    }

    fn reap_rx(&mut self) {
        let used = self.memory[RX_USED as usize..RX_USED as usize + 4 + 4 * USED_ELEM_LEN].to_vec();
        self.rx
            .reap(&mut self.memory[RX_TABLE as usize..], &used)
            .expect("rx reap")
            .expect("rx completion published");
    }
}

fn batches(iterations: u64) -> impl Iterator<Item = usize> {
    let mut remaining = iterations;
    std::iter::from_fn(move || {
        (remaining > 0).then(|| {
            let count = remaining.min(BATCH as u64);
            remaining -= count;
            count as usize
        })
    })
}

fn tx_chain_drain(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("virtio_net/tx_chain_to_frame");
    group.throughput(Throughput::Elements(1));
    for size in SIZES {
        let frame = vec![0xA5_u8; size];
        let (mut transport, from_guest, _to_guest) = wired_transport();
        let mut guest = Guest::new();
        let id = BenchmarkId::new(format!("{NEUTRAL}/drain_tx"), size);
        group.bench_function(id, |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for count in batches(iterations) {
                    for slot in 0..count {
                        guest.transmit(slot, &frame);
                    }
                    let started = Instant::now();
                    let drained = transport.drain_tx(TX_QUEUE, &mut guest.memory).expect("drains");
                    black_box(&drained);
                    drop(drained);
                    total += started.elapsed();
                    for _ in 0..count {
                        from_guest.call(()).now_or_never().expect("forwarded").expect("open");
                        guest.reap_tx();
                    }
                }
                total
            });
        });
        let mut floor_memory = vec![0_u8; NET_HDR_LEN + size];
        let source = vec![0xA5_u8; NET_HDR_LEN + size];
        let id = BenchmarkId::new(format!("{NEUTRAL}/floor_copy_from_slice"), size);
        group.bench_function(id, |bencher| {
            bencher.iter(|| {
                floor_memory.copy_from_slice(black_box(&source));
                black_box(&floor_memory);
            });
        });
    }
    group.finish();
}

fn rx_fill(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("virtio_net/frame_to_rx_chain");
    group.throughput(Throughput::Elements(1));
    for size in SIZES {
        let frame = vec![0xA5_u8; size];
        let (mut transport, _from_guest, to_guest) = wired_transport();
        let mut guest = Guest::new();
        let id = BenchmarkId::new(format!("{NEUTRAL}/fill_rx"), size);
        group.bench_function(id, |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for count in batches(iterations) {
                    for slot in 0..count {
                        guest.post_rx_buffer(slot);
                        to_guest.call(frame.clone()).now_or_never().expect("ready").expect("room");
                    }
                    let started = Instant::now();
                    let delivered = transport.fill_rx(&mut guest.memory).expect("fills");
                    total += started.elapsed();
                    assert_eq!(delivered, count, "every posted buffer must take a frame");
                    for _ in 0..count {
                        guest.reap_rx();
                    }
                }
                total
            });
        });
        let mut floor_memory = vec![0_u8; NET_HDR_LEN + size];
        let source = vec![0xA5_u8; NET_HDR_LEN + size];
        let id = BenchmarkId::new(format!("{NEUTRAL}/floor_copy_from_slice"), size);
        group.bench_function(id, |bencher| {
            bencher.iter(|| {
                floor_memory.copy_from_slice(black_box(&source));
                black_box(&floor_memory);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, tx_chain_drain, rx_fill);
criterion_main!(benches);
