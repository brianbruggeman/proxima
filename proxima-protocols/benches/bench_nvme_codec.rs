//! Compare the borrowed-view sans-IO codec against the conventional owned
//! mirror-struct codec — a driver materialises an owned `NvmeCommand` /
//! `NvmeCompletion` struct (the `borrow_before_own` shape SPDK's
//! `struct spdk_nvme_cmd` / `spdk_nvme_cpl` and the registry `nvme` crates use),
//! then reads its fields. This crate ships zero unsafe and the workspace has no
//! safe-transmute crate (no zerocopy/bytemuck), so the incumbent is modelled the
//! way proxima would actually write it: `from_le_bytes` field reads, not a raw
//! pointer cast. The borrowed view must tie or beat the eager owned mirror on
//! the per-IOP hot paths (submit encode, completion reap).
#![allow(clippy::expect_used)]

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::nvme::{
    CommandBuilder, CompletionEntry, CompletionRing, StatusField, SubmissionEntry,
    command::ENTRY_LEN as SQE_LEN, completion::ENTRY_LEN as CQE_LEN,
};
use std::hint::black_box;

const OPC_READ: u8 = 0x02;
const BURST: usize = 1024;

// The incumbent: an owned NVMe command struct a driver populates then serialises
// into the submission ring, field by field.
#[derive(Clone, Copy, Default)]
struct OwnedCommand {
    opc_flags_cid: u32,
    nsid: u32,
    mptr: u64,
    prp1: u64,
    prp2: u64,
    command_dwords: [u32; 6],
}

// the unused mirror fields are the point: an eager owned codec copies all four
// dwords out, then reads only the one it needs — that full copy is the cost we
// bench against the lazy borrowed view.
#[allow(dead_code)]
#[derive(Clone, Copy, Default)]
struct OwnedCompletion {
    cdw0: u32,
    cdw1: u32,
    sqhd_sqid: u32,
    cid_status: u32,
}

impl OwnedCompletion {
    // eager owned mirror: copy every dword out of the slot, then read fields —
    // the borrow_before_own shape, the opposite of the lazy borrowed view.
    fn from_slot(slot: &[u8]) -> Self {
        Self {
            cdw0: u32::from_le_bytes([slot[0], slot[1], slot[2], slot[3]]),
            cdw1: u32::from_le_bytes([slot[4], slot[5], slot[6], slot[7]]),
            sqhd_sqid: u32::from_le_bytes([slot[8], slot[9], slot[10], slot[11]]),
            cid_status: u32::from_le_bytes([slot[12], slot[13], slot[14], slot[15]]),
        }
    }
}

fn build_read_sqe(buffer: &mut [u8], slba: u64) {
    CommandBuilder::new(OPC_READ, 0x0007)
        .namespace_id(1)
        .data_ptrs(0xdead_beef_0000_1000, 0)
        .command_dword(0, slba as u32)
        .command_dword(1, (slba >> 32) as u32)
        .command_dword(2, 7)
        .write(buffer)
        .expect("64-byte buffer");
}

fn build_read_owned(buffer: &mut [u8], slba: u64) {
    let mut command = OwnedCommand {
        opc_flags_cid: u32::from(OPC_READ) | (0x0007u32 << 16),
        nsid: 1,
        prp1: 0xdead_beef_0000_1000,
        ..Default::default()
    };
    command.command_dwords[0] = slba as u32;
    command.command_dwords[1] = (slba >> 32) as u32;
    command.command_dwords[2] = 7;

    let slot = &mut buffer[..SQE_LEN];
    slot[0..4].copy_from_slice(&command.opc_flags_cid.to_le_bytes());
    slot[4..8].copy_from_slice(&command.nsid.to_le_bytes());
    slot[8..16].fill(0);
    slot[16..24].copy_from_slice(&command.mptr.to_le_bytes());
    slot[24..32].copy_from_slice(&command.prp1.to_le_bytes());
    slot[32..40].copy_from_slice(&command.prp2.to_le_bytes());
    for (index, dword) in command.command_dwords.iter().enumerate() {
        slot[40 + index * 4..44 + index * 4].copy_from_slice(&dword.to_le_bytes());
    }
}

fn sqe_encode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sqe_encode");

    group.throughput(Throughput::Elements(1));
    group.bench_function("single/builder", |bencher| {
        let mut buffer = [0u8; SQE_LEN];
        bencher.iter(|| build_read_sqe(black_box(&mut buffer), black_box(0x1234_5678)));
    });
    group.bench_function("single/owned_struct", |bencher| {
        let mut buffer = [0u8; SQE_LEN];
        bencher.iter(|| build_read_owned(black_box(&mut buffer), black_box(0x1234_5678)));
    });

    group.throughput(Throughput::Elements(BURST as u64));
    group.bench_function("burst1024/builder", |bencher| {
        let mut ring = vec![0u8; SQE_LEN * BURST];
        bencher.iter(|| {
            for (index, slot) in ring.chunks_exact_mut(SQE_LEN).enumerate() {
                build_read_sqe(slot, index as u64);
            }
            black_box(&ring);
        });
    });
    group.bench_function("burst1024/owned_struct", |bencher| {
        let mut ring = vec![0u8; SQE_LEN * BURST];
        bencher.iter(|| {
            for (index, slot) in ring.chunks_exact_mut(SQE_LEN).enumerate() {
                build_read_owned(slot, index as u64);
            }
            black_box(&ring);
        });
    });

    group.finish();
}

// The per-IOP completion reap: read phase (is this slot fresh?), CID (which
// command finished), and success. This is the 100%-frequency 80% case.
fn reap_borrowed(slot: &[u8]) -> (bool, u16, bool) {
    let view = CompletionEntry::parse(slot).expect("16-byte slot");
    (view.phase(), view.command_id(), view.status().is_success())
}

// folded reap: CID and status come from the single dword 3 the controller
// writes, matching the struct-cast's one-load shape.
fn reap_borrowed_folded(slot: &[u8]) -> (bool, u16, bool) {
    let view = CompletionEntry::parse(slot).expect("16-byte slot");
    let (cid, status) = view.command_id_and_status();
    (status.phase(), cid, status.is_success())
}

fn reap_owned(slot: &[u8]) -> (bool, u16, bool) {
    let completion = OwnedCompletion::from_slot(slot);
    let cid_status = completion.cid_status;
    let cid = (cid_status & 0xffff) as u16;
    let status = (cid_status >> 16) as u16;
    (status & 1 != 0, cid, (status >> 1) & 0x7ff == 0)
}

fn cqe_decode(criterion: &mut Criterion) {
    let mut completion = [0u8; CQE_LEN];
    proxima_protocols::nvme::write_completion(
        &mut completion,
        0x1234_5678,
        0x0042,
        0x0001,
        0x00ab,
        StatusField::from_bits(0x0001),
    )
    .expect("16-byte buffer");

    let mut group = criterion.benchmark_group("cqe_decode");
    group.throughput(Throughput::Elements(1));
    group.bench_function("single/borrowed_view", |bencher| {
        bencher.iter(|| black_box(reap_borrowed(black_box(&completion))));
    });
    group.bench_function("single/borrowed_folded", |bencher| {
        bencher.iter(|| black_box(reap_borrowed_folded(black_box(&completion))));
    });
    group.bench_function("single/owned_struct", |bencher| {
        bencher.iter(|| black_box(reap_owned(black_box(&completion))));
    });

    // a full completion-queue drain: BURST fresh CQEs reaped in one poll.
    let mut ring = vec![0u8; CQE_LEN * BURST];
    for (index, slot) in ring.chunks_exact_mut(CQE_LEN).enumerate() {
        proxima_protocols::nvme::write_completion(
            slot,
            0,
            0,
            0,
            index as u16,
            StatusField::from_bits(0x0001),
        )
        .expect("16-byte slot");
    }
    group.throughput(Throughput::Elements(BURST as u64));
    group.bench_function("burst1024/borrowed_view", |bencher| {
        bencher.iter(|| {
            let mut acc = 0u32;
            for slot in ring.chunks_exact(CQE_LEN) {
                let (phase, cid, ok) = reap_borrowed(slot);
                acc = acc.wrapping_add(u32::from(phase) + u32::from(cid) + u32::from(ok));
            }
            black_box(acc);
        });
    });
    group.bench_function("burst1024/borrowed_folded", |bencher| {
        bencher.iter(|| {
            let mut acc = 0u32;
            for slot in ring.chunks_exact(CQE_LEN) {
                let (phase, cid, ok) = reap_borrowed_folded(slot);
                acc = acc.wrapping_add(u32::from(phase) + u32::from(cid) + u32::from(ok));
            }
            black_box(acc);
        });
    });
    group.bench_function("burst1024/owned_struct", |bencher| {
        bencher.iter(|| {
            let mut acc = 0u32;
            for slot in ring.chunks_exact(CQE_LEN) {
                let (phase, cid, ok) = reap_owned(slot);
                acc = acc.wrapping_add(u32::from(phase) + u32::from(cid) + u32::from(ok));
            }
            black_box(acc);
        });
    });
    group.finish();
}

// The ring FSM steady state: poll-and-advance a completion cursor. No incumbent
// (pure index/phase arithmetic) — neutral noise-floor arm.
fn ring_poll(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("ring_poll");
    group.throughput(Throughput::Elements(1));
    group.bench_function("completion_advance", |bencher| {
        bencher.iter_batched(
            || CompletionRing::new(1024).expect("legal depth"),
            |mut ring| {
                let fresh = ring.is_ready(black_box(ring.expected_phase()));
                let head = ring.advance();
                black_box((fresh, head));
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

// the submission decode path: parse a slot and read the fields a controller
// model would inspect, proving decode parity alongside the encode arms.
fn sqe_decode_roundtrip(criterion: &mut Criterion) {
    let mut sqe = [0u8; SQE_LEN];
    build_read_sqe(&mut sqe, 0x1234_5678);
    let mut group = criterion.benchmark_group("sqe_decode");
    group.throughput(Throughput::Elements(1));
    group.bench_function("single/borrowed_view", |bencher| {
        bencher.iter(|| {
            let view = SubmissionEntry::parse(black_box(&sqe)).expect("64-byte slot");
            black_box((view.opcode(), view.command_id(), view.command_dword(0)));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    sqe_encode,
    cqe_decode,
    sqe_decode_roundtrip,
    ring_poll
);
criterion_main!(benches);
