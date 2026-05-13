#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! P6 — memcached text-protocol parser bench. Two arms:
//!
//! - `proxima` — `proxima::memcached::parse_command`.
//! - `parity` — hand-rolled equivalent inline; same scope (returns
//!   the same borrowed slices and parsed integers).
//!
//! Three workloads:
//!
//! - `get` — `get mykey\r\n` (most common read op).
//! - `set` — `set mykey 0 60 64\r\n<64 bytes>\r\n` (common write op).
//! - `delete` — `delete mykey\r\n` (common evict op).
//!
//! `memcached-rs` was named as the reference crate in
//! `docs/protocol-gap/discipline.md`, but it's a client lib without
//! a separate parser module — same scope-mismatched problem we hit
//! with P5/Redis. The hand-rolled parity baseline is the
//! gate-passing comparison.
//!
//! incumbents:
//!   - (none — no maintained server-side memcached text-protocol
//!     parser crate in the Rust ecosystem at the scope proxima covers)
//!
//! groups (and design-favors per workload):
//!   - memcached_get / memcached_set / memcached_delete   design-favors: proxima
//!     (parity baseline only; no incumbent available)
//!
//! REGIME OUT-OF-SCOPE: no equivalent-scope incumbent crate exists.
//! Comparison defers to the hand-rolled parity reference; gate-13 is
//! satisfied via the documented absence of a real incumbent rather
//! than a measured win on home turf. See protocol-gap discipline.md.

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use proxima_protocols::memcached::{Command, StoreMode, parse_command as proxima_parse};

/// Same shape as `proxima::memcached::Command` — the bench measures
/// parser shape cost not output struct difference.
#[allow(dead_code)]
enum ParityCommand<'a> {
    Get {
        keys: &'a [u8],
        gets: bool,
    },
    Set {
        key: &'a [u8],
        flags: u32,
        exptime: u32,
        value: &'a [u8],
        noreply: bool,
    },
    Delete {
        key: &'a [u8],
        noreply: bool,
    },
}

fn parity_parse(buf: &[u8]) -> Option<(ParityCommand<'_>, usize)> {
    let line_end = find_crlf(buf)?;
    let line = &buf[..line_end];
    let after_line = line_end + 2;
    let (verb, rest) = split_token(line);
    // proxima dispatches on 14 verbs (get/gets/set/add/replace/append/prepend/
    // cas/delete/incr/decr/touch/flush_all/stats/version/quit). To make the
    // bench apples-to-apples, parity must do the same wide-match dispatch
    // even if only get/set/delete are implemented — otherwise parity gets
    // an unfair narrower-table branch-prediction advantage.
    match verb {
        b"get" | b"gets" => Some((
            ParityCommand::Get {
                keys: rest,
                gets: verb == b"gets",
            },
            after_line,
        )),
        b"add" | b"replace" | b"append" | b"prepend" | b"cas" | b"incr" | b"decr" | b"touch"
        | b"flush_all" | b"stats" | b"version" | b"quit" => None,
        b"set" => {
            let (key, rest) = split_token(rest);
            let (flags_tok, rest) = split_token(rest);
            let (exptime_tok, rest) = split_token(rest);
            let (bytes_tok, rest) = split_token(rest);
            let flags = parse_u32(flags_tok)?;
            let exptime = parse_u32(exptime_tok)?;
            let value_len = parse_u32(bytes_tok)? as usize;
            let noreply = rest.starts_with(b"noreply");
            let value_start = after_line;
            let value_end = value_start + value_len;
            if buf.len() < value_end + 2 {
                return None;
            }
            Some((
                ParityCommand::Set {
                    key,
                    flags,
                    exptime,
                    value: &buf[value_start..value_end],
                    noreply,
                },
                value_end + 2,
            ))
        }
        b"delete" => {
            let (key, rest) = split_token(rest);
            Some((
                ParityCommand::Delete {
                    key,
                    noreply: rest.starts_with(b"noreply"),
                },
                after_line,
            ))
        }
        _ => None,
    }
}

#[inline]
fn find_crlf(buf: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i + 1 < buf.len() {
        if buf[i] == b'\r' && buf[i + 1] == b'\n' {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[inline]
fn split_token(line: &[u8]) -> (&[u8], &[u8]) {
    match line.iter().position(|&b| b == b' ') {
        Some(idx) => {
            let mut rest_start = idx + 1;
            while rest_start < line.len() && line[rest_start] == b' ' {
                rest_start += 1;
            }
            (&line[..idx], &line[rest_start..])
        }
        None => (line, &[]),
    }
}

#[inline]
fn parse_u32(token: &[u8]) -> Option<u32> {
    if token.is_empty() {
        return None;
    }
    let mut value: u32 = 0;
    for &byte in token {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?.checked_add(u32::from(byte - b'0'))?;
    }
    Some(value)
}

const GET_REQUEST: &[u8] = b"get mykey\r\n";
const DELETE_REQUEST: &[u8] = b"delete mykey\r\n";

fn make_set_request() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"set mykey 0 60 64\r\n");
    buf.extend_from_slice(&[0xAB; 64]);
    buf.extend_from_slice(b"\r\n");
    buf
}

fn bench_get(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("memcached_get");
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Bytes(GET_REQUEST.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let (cmd, used) = proxima_parse(std::hint::black_box(GET_REQUEST)).unwrap();
            // black_box the keys slice to keep the match arm alive.
            match cmd {
                Command::Get { keys, .. } => std::hint::black_box(keys),
                _ => unreachable!(),
            };
            std::hint::black_box(used);
        });
    });
    group.bench_function("parity", |bencher| {
        bencher.iter(|| {
            let (cmd, used) = parity_parse(std::hint::black_box(GET_REQUEST)).unwrap();
            match cmd {
                ParityCommand::Get { keys, .. } => std::hint::black_box(keys),
                _ => unreachable!(),
            };
            std::hint::black_box(used);
        });
    });
    group.finish();
}

fn bench_set(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("memcached_set");
    group.measurement_time(Duration::from_secs(2));
    let set_request = make_set_request();
    group.throughput(Throughput::Bytes(set_request.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let (cmd, used) = proxima_parse(std::hint::black_box(&set_request)).unwrap();
            match cmd {
                Command::Store {
                    mode, key, value, ..
                } => {
                    assert_eq!(mode, StoreMode::Set);
                    std::hint::black_box((key, value));
                }
                _ => unreachable!(),
            };
            std::hint::black_box(used);
        });
    });
    group.bench_function("parity", |bencher| {
        bencher.iter(|| {
            let (cmd, used) = parity_parse(std::hint::black_box(&set_request)).unwrap();
            match cmd {
                ParityCommand::Set { key, value, .. } => std::hint::black_box((key, value)),
                _ => unreachable!(),
            };
            std::hint::black_box(used);
        });
    });
    group.finish();
}

fn bench_delete(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("memcached_delete");
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Bytes(DELETE_REQUEST.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let (cmd, used) = proxima_parse(std::hint::black_box(DELETE_REQUEST)).unwrap();
            match cmd {
                Command::Delete { key, .. } => std::hint::black_box(key),
                _ => unreachable!(),
            };
            std::hint::black_box(used);
        });
    });
    group.bench_function("parity", |bencher| {
        bencher.iter(|| {
            let (cmd, used) = parity_parse(std::hint::black_box(DELETE_REQUEST)).unwrap();
            match cmd {
                ParityCommand::Delete { key, .. } => std::hint::black_box(key),
                _ => unreachable!(),
            };
            std::hint::black_box(used);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_get, bench_set, bench_delete);
criterion_main!(benches);
