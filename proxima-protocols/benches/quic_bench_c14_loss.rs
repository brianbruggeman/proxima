//! C14 — loss detection bench arms.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_protocols::quic::loss::{LossDetection, SentPacket};
use proxima_protocols::quic::time::{Duration, Instant};
use proxima_protocols::quic::tls::Epoch;

fn sent(pn: u64, sent_time: Instant) -> SentPacket {
    SentPacket {
        packet_number: pn,
        sent_time,
        size_bytes: 1200,
        is_ack_eliciting: true,
        in_flight: true,
    }
}

fn bench_on_packet_sent(criterion: &mut Criterion) {
    criterion.bench_function("c14_on_packet_sent_1024", |bencher| {
        bencher.iter(|| {
            let mut det = LossDetection::new();
            for pn in 0..1024u64 {
                det.on_packet_sent(
                    Epoch::Application,
                    sent(black_box(pn), Instant::from_micros(1_000_000 + pn * 100)),
                );
            }
            black_box(det);
        });
    });
}

fn bench_rtt_sample(criterion: &mut Criterion) {
    criterion.bench_function("c14_rtt_on_sample_1000", |bencher| {
        bencher.iter(|| {
            let mut det = LossDetection::new();
            for _ in 0..1000u64 {
                det.rtt
                    .on_sample(black_box(Duration::from_millis(50)), Duration::ZERO);
            }
            black_box(det.rtt);
        });
    });
}

fn bench_detect_losses_packet_threshold(criterion: &mut Criterion) {
    criterion.bench_function("c14_detect_losses_packet_threshold_1000", |bencher| {
        bencher.iter_batched(
            || {
                let mut det = LossDetection::new();
                for pn in 0..1000u64 {
                    det.on_packet_sent(
                        Epoch::Application,
                        sent(pn, Instant::from_micros(1_000_000 + pn * 100)),
                    );
                }
                det
            },
            |mut det| {
                let outcome = det.on_ack_received(
                    Epoch::Application,
                    999,
                    Duration::ZERO,
                    &[(999, 999)],
                    Instant::from_micros(2_000_000),
                );
                black_box(outcome);
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_compute_pto(criterion: &mut Criterion) {
    criterion.bench_function("c14_compute_pto", |bencher| {
        let mut det = LossDetection::new();
        det.rtt
            .on_sample(Duration::from_millis(100), Duration::ZERO);
        bencher.iter(|| {
            let pto = det.compute_pto(true);
            black_box(pto);
        });
    });
}

fn bench_discard_epoch(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c14_discard_epoch");
    // Multi-size sweep per /disciplined-component point 9.
    for size in [0u64, 16, 256, 1024] {
        group.bench_function(format!("size_{size}"), |bencher| {
            bencher.iter_batched(
                || {
                    let mut det = LossDetection::new();
                    for pn in 0..size {
                        det.on_packet_sent(
                            Epoch::Handshake,
                            sent(pn, Instant::from_micros(1_000_000 + pn * 100)),
                        );
                    }
                    det
                },
                |mut det| {
                    let released = det.discard_epoch(black_box(Epoch::Handshake));
                    black_box(released);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_next_deadline(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c14_next_deadline");
    // Empty: no epoch armed (returns None — exits early).
    group.bench_function("empty", |bencher| {
        let det = LossDetection::new();
        bencher.iter(|| {
            black_box(det.next_deadline());
        });
    });
    // Single epoch armed: typical Application-only steady state.
    group.bench_function("single_epoch_armed", |bencher| {
        let mut det = LossDetection::new();
        det.on_packet_sent(Epoch::Application, sent(0, Instant::from_micros(1_000_000)));
        bencher.iter(|| {
            black_box(det.next_deadline());
        });
    });
    // All three epochs armed: worst-case (Initial + Handshake + Application
    // all have outstanding ack-eliciting packets — the .min() walks 3 PTO
    // computations + the same for loss_time).
    group.bench_function("three_epochs_armed", |bencher| {
        let mut det = LossDetection::new();
        det.on_packet_sent(Epoch::Initial, sent(0, Instant::from_micros(1_000_000)));
        det.on_packet_sent(Epoch::Handshake, sent(0, Instant::from_micros(1_100_000)));
        det.on_packet_sent(Epoch::Application, sent(0, Instant::from_micros(1_200_000)));
        bencher.iter(|| {
            black_box(det.next_deadline());
        });
    });
    group.finish();
}

fn bench_on_loss_detection_timeout(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c14_on_loss_detection_timeout");
    // PTO path: no loss_time set; timeout bumps pto_count and selects
    // an epoch. The hot path in handle_timeout.
    group.bench_function("pto_path", |bencher| {
        bencher.iter_batched(
            || {
                let mut det = LossDetection::new();
                det.on_packet_sent(Epoch::Application, sent(0, Instant::from_micros(1_000_000)));
                det
            },
            |mut det| {
                let outcome =
                    det.on_loss_detection_timeout(black_box(Instant::from_micros(10_000_000)));
                black_box(outcome);
            },
            criterion::BatchSize::SmallInput,
        );
    });
    // Loss-threshold path: loss_time armed; timeout walks detect_losses
    // for the earliest-armed epoch. Typical ACK-driven processing.
    group.bench_function("loss_threshold_path_100_in_flight", |bencher| {
        bencher.iter_batched(
            || {
                let mut det = LossDetection::new();
                for pn in 0..100u64 {
                    det.on_packet_sent(
                        Epoch::Application,
                        sent(pn, Instant::from_micros(1_000_000 + pn * 100)),
                    );
                }
                let _ = det.on_ack_received(
                    Epoch::Application,
                    99,
                    Duration::ZERO,
                    &[(99, 99)],
                    Instant::from_micros(1_500_000),
                );
                det
            },
            |mut det| {
                let outcome =
                    det.on_loss_detection_timeout(black_box(Instant::from_micros(2_000_000)));
                black_box(outcome);
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_on_packet_sent,
    bench_rtt_sample,
    bench_detect_losses_packet_threshold,
    bench_compute_pto,
    bench_discard_epoch,
    bench_next_deadline,
    bench_on_loss_detection_timeout,
);
criterion_main!(benches);
