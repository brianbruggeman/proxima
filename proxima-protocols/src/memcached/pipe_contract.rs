//! The memcached-over-`Pipe` contract: how a memcached command maps onto a
//! `proxima_primitives::pipe::Pipe` request, and what rides back.
//!
//! This is the RISC payoff (workspace principle 1) — `proxima-memcached`
//! does not own a bespoke client trait, it speaks the one workspace
//! primitive, `Pipe`, the way `proxima-redis` does. [`MemcachedRequest`] is
//! the owned, `'static` mirror of [`Command`] (the same role
//! `crate::redis::RespValue::from_frame` plays for a borrowed
//! [`crate::redis::Frame`]): a business handler pipe's `Request.payload`
//! carries a whole [`MemcachedRequest`], fully typed — no downcast, no
//! arg-bag the handler has to re-parse per verb shape.
//!
//! # Zero-copy re-owning (workspace principles 1, 11)
//!
//! `key`/`value`/`args` are [`Bytes`] windows sliced from the same backing
//! buffer the wire command was parsed from via [`Bytes::slice_ref`] — an
//! `Arc` refcount bump, not a copy — mirroring the pattern
//! `grpc_framing::frame_codec_pipe`/`http1_codec::frame_codec_pipe`/
//! `websocket_frame::frame_codec_pipe` already ship on the same
//! `codec_pipe::OwnFrame` seam.
//!
//! A multi-`get`'s keys are NOT materialized into any container at all.
//! `MemcachedRequest::Get::keys` is ONE `Bytes` — the untouched
//! `"k1 k2 k3"` span, still space-joined exactly as it arrived on the
//! wire — because every owned collection (`Vec`, `Box`, `ArrayVec`,
//! `heapless::Vec`) allocates, and a fixed-cap container additionally taxes
//! the SAME allocation cost onto every request shape sharing its enum
//! (measured: `Box<ArrayVec<Bytes, 64>>` made every `MemcachedRequest`
//! value pay for a 64-slot allocation, regressing the single-key `get`
//! path by >100%; see git history for that dead end). [`iter_keys`] walks
//! the span lazily, splitting on `b' '` via `memchr::memchr` (workspace
//! principle 11 — SIMD byte scan) one pass, zero allocation. A single key
//! is just a span the iterator walks once — there is no separate
//! single-key/multi-key code path, and no cap to enforce: the DoS bound is
//! [`super::frame_codec::MemcachedCodec::max_message_bytes`] at
//! `parse_frame` (the whole command, keys span included, must already fit
//! before a `Command::Get` is ever produced).
//!
//! This tier can never be alloc-free (`Bytes` is `Arc`-backed by
//! construction) — the claim is O(payload) copied → O(1) re-owned (one
//! `Arc` refcount bump, already paid once per request, not per key), on
//! this alloc tier, not zero-alloc and not the no-alloc floor. The bare
//! no_std FSM tier ([`super::connection`]) is the genuine zero-alloc
//! floor: borrowed `Command<'a>` in, borrowed `Command<'a>` out.
//!
//! # Buffer genericity (component C4)
//!
//! [`MemcachedRequest`] is generic over `T: ShareBuf`, defaulted to
//! [`Bytes`] — a caller drives the codec over their OWN buffer type (an
//! `Arc<[u8]>`-backed window, a DPDK `rte_mbuf`, ...) with no behavior or
//! allocation change on the `T = Bytes` path. `share` (not
//! `Bytes::slice_ref`) is the re-owning primitive throughout this module;
//! see `proxima_codec::share_buf`'s own doc for why the seam is `share`
//! and not `bytes::Buf`.

use alloc::vec::Vec;

use bytes::Bytes;
use proxima_codec::ShareBuf;

use super::{Command, StoreMode};

/// Command verbs a caller sets as `Request.method` (uppercased, mirroring
/// `crate::redis::pipe_contract::verb`'s convention of a symbolic routing
/// label — NOT the literal lowercase wire spelling; see
/// [`encode_request`] for that).
pub mod verb {
    pub const GET: &str = "get";
    pub const GETS: &str = "gets";
    pub const SET: &str = "set";
    pub const ADD: &str = "add";
    pub const REPLACE: &str = "replace";
    pub const APPEND: &str = "append";
    pub const PREPEND: &str = "prepend";
    pub const CAS: &str = "cas";
    pub const DELETE: &str = "delete";
    pub const INCR: &str = "incr";
    pub const DECR: &str = "decr";
    pub const TOUCH: &str = "touch";
    pub const FLUSH_ALL: &str = "flush_all";
    pub const STATS: &str = "stats";
    pub const VERSION: &str = "version";
    pub const QUIT: &str = "quit";
}

/// The typed payload a caller's `Request.payload` carries — the owned,
/// `'static` mirror of [`Command`], one variant per memcached command
/// shape. Generic over `T: ShareBuf` (component C4), defaulted to
/// [`Bytes`] — see the module doc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemcachedRequest<T = Bytes> {
    Get {
        /// The untouched, still space-joined `"k1 k2 k3"` wire span —
        /// see the module doc. Walk it with [`iter_keys`].
        keys: T,
        gets: bool,
    },
    Store {
        mode: StoreMode,
        key: T,
        flags: u32,
        exptime: u32,
        value: T,
        noreply: bool,
    },
    Cas {
        key: T,
        flags: u32,
        exptime: u32,
        cas_unique: u64,
        value: T,
        noreply: bool,
    },
    Delete {
        key: T,
        noreply: bool,
    },
    Counter {
        increment: bool,
        key: T,
        delta: u64,
        noreply: bool,
    },
    Touch {
        key: T,
        exptime: u32,
        noreply: bool,
    },
    FlushAll {
        delay: Option<u32>,
        noreply: bool,
    },
    Stats {
        args: T,
    },
    Version,
    Quit,
}

impl<T: ShareBuf> MemcachedRequest<T> {
    /// Lift a borrowed, parse-buffer-tied [`Command`] into an owned,
    /// `'static` request — the async-boundary conversion a handler pipe
    /// needs (the borrowed form cannot outlive the connection's read
    /// buffer past the point the driver calls `Connection::consume`).
    /// `source` must be the exact `T` window `command` was parsed from
    /// (`command`'s slices become `source.share(..)` windows into it — an
    /// `Arc` refcount bump on the `T = Bytes` default, not a copy).
    #[must_use]
    pub fn from_command(source: &T, command: &Command<'_>) -> Self {
        // `Command<'_>`'s fields are all Copy-eligible (`&[u8]`/`u32`/`bool`);
        // cloning is a bitwise copy of borrowed slices, not an allocation —
        // matching on the clone (by value) avoids the `&&[u8]` double
        // indirection a reference-pattern match ergonomics would produce.
        match command.clone() {
            Command::Get { keys, gets } => Self::Get {
                keys: source.share(keys),
                gets,
            },
            Command::Store {
                mode,
                key,
                flags,
                exptime,
                value,
                noreply,
            } => Self::Store {
                mode,
                key: source.share(key),
                flags,
                exptime,
                value: source.share(value),
                noreply,
            },
            Command::Cas {
                key,
                flags,
                exptime,
                cas_unique,
                value,
                noreply,
            } => Self::Cas {
                key: source.share(key),
                flags,
                exptime,
                cas_unique,
                value: source.share(value),
                noreply,
            },
            Command::Delete { key, noreply } => Self::Delete {
                key: source.share(key),
                noreply,
            },
            Command::Counter {
                increment,
                key,
                delta,
                noreply,
            } => Self::Counter {
                increment,
                key: source.share(key),
                delta,
                noreply,
            },
            Command::Touch {
                key,
                exptime,
                noreply,
            } => Self::Touch {
                key: source.share(key),
                exptime,
                noreply,
            },
            Command::FlushAll { delay, noreply } => Self::FlushAll { delay, noreply },
            Command::Stats { args } => Self::Stats {
                args: source.share(args),
            },
            Command::Version => Self::Version,
            Command::Quit => Self::Quit,
        }
    }

    /// Whether the real server suppresses any reply for this command —
    /// `true` only for the storage/mutation family's own `noreply` flag;
    /// `get`/`stats`/`version`/`quit` have no such concept in the wire
    /// grammar.
    #[must_use]
    pub fn is_noreply(&self) -> bool {
        match self {
            Self::Store { noreply, .. }
            | Self::Cas { noreply, .. }
            | Self::Delete { noreply, .. }
            | Self::Counter { noreply, .. }
            | Self::Touch { noreply, .. }
            | Self::FlushAll { noreply, .. } => *noreply,
            Self::Get { .. } | Self::Stats { .. } | Self::Version | Self::Quit => false,
        }
    }
}

/// Walks a `Get`'s `keys` span (`"k1 k2 k3"`), yielding each key as a `T`
/// sub-slice — an `Arc` refcount bump per key on the `T = Bytes` default,
/// not a copy. Splits lazily on `b' '` via [`memchr::memchr`] (workspace
/// principle 11: SIMD byte scan), one pass, zero allocation: no
/// `Vec`/`Box`/`ArrayVec` materializes the key list. A run of consecutive
/// spaces (or a leading one) yields no empty keys, matching the wire
/// grammar's tokenization.
pub fn iter_keys<T: ShareBuf>(keys: &T) -> impl Iterator<Item = T> + '_ {
    let mut remaining = keys.clone();
    core::iter::from_fn(move || {
        loop {
            if remaining.is_empty() {
                return None;
            }
            match memchr::memchr(b' ', &remaining) {
                Some(0) => remaining = remaining.share(&remaining[1..]),
                Some(index) => {
                    let key = remaining.share(&remaining[..index]);
                    remaining = remaining.share(&remaining[index + 1..]);
                    return Some(key);
                }
                None => {
                    let key = remaining.share(&remaining[..]);
                    remaining = remaining.share(&remaining[remaining.len()..]);
                    return Some(key);
                }
            }
        }
    })
}

fn store_verb(mode: StoreMode) -> &'static [u8] {
    match mode {
        StoreMode::Set => b"set",
        StoreMode::Add => b"add",
        StoreMode::Replace => b"replace",
        StoreMode::Append => b"append",
        StoreMode::Prepend => b"prepend",
    }
}

fn write_decimal_u32(dest: &mut Vec<u8>, value: u32) {
    write_decimal_u64(dest, u64::from(value));
}

fn write_decimal_u64(dest: &mut Vec<u8>, mut value: u64) {
    let mut digits = [0_u8; 20];
    let mut count = 0;
    loop {
        digits[count] = b'0' + (value % 10) as u8;
        value /= 10;
        count += 1;
        if value == 0 {
            break;
        }
    }
    dest.extend(digits[..count].iter().rev());
}

fn write_noreply(dest: &mut Vec<u8>, noreply: bool) {
    if noreply {
        dest.extend_from_slice(b" noreply");
    }
}

/// Encode a [`MemcachedRequest`] as the wire command [`super::parse_command`]
/// accepts back — the client's outbound path. Not routed through
/// [`Command`] (there is no existing request encoder to reuse).
/// `Get::keys` is already the space-joined wire span, so it is written
/// once, verbatim — the SAME shape [`Command::Get::keys`] itself is.
pub fn encode_request<T: ShareBuf>(request: &MemcachedRequest<T>, dest: &mut Vec<u8>) {
    match request {
        MemcachedRequest::Get { keys, gets } => {
            dest.extend_from_slice(if *gets { b"gets" } else { b"get" });
            if !keys.is_empty() {
                dest.push(b' ');
                dest.extend_from_slice(keys);
            }
            dest.extend_from_slice(b"\r\n");
        }
        MemcachedRequest::Store {
            mode,
            key,
            flags,
            exptime,
            value,
            noreply,
        } => {
            dest.extend_from_slice(store_verb(*mode));
            dest.push(b' ');
            dest.extend_from_slice(key);
            dest.push(b' ');
            write_decimal_u32(dest, *flags);
            dest.push(b' ');
            write_decimal_u32(dest, *exptime);
            dest.push(b' ');
            write_decimal_u32(dest, value.len() as u32);
            write_noreply(dest, *noreply);
            dest.extend_from_slice(b"\r\n");
            dest.extend_from_slice(value);
            dest.extend_from_slice(b"\r\n");
        }
        MemcachedRequest::Cas {
            key,
            flags,
            exptime,
            cas_unique,
            value,
            noreply,
        } => {
            dest.extend_from_slice(b"cas ");
            dest.extend_from_slice(key);
            dest.push(b' ');
            write_decimal_u32(dest, *flags);
            dest.push(b' ');
            write_decimal_u32(dest, *exptime);
            dest.push(b' ');
            write_decimal_u32(dest, value.len() as u32);
            dest.push(b' ');
            write_decimal_u64(dest, *cas_unique);
            write_noreply(dest, *noreply);
            dest.extend_from_slice(b"\r\n");
            dest.extend_from_slice(value);
            dest.extend_from_slice(b"\r\n");
        }
        MemcachedRequest::Delete { key, noreply } => {
            dest.extend_from_slice(b"delete ");
            dest.extend_from_slice(key);
            write_noreply(dest, *noreply);
            dest.extend_from_slice(b"\r\n");
        }
        MemcachedRequest::Counter {
            increment,
            key,
            delta,
            noreply,
        } => {
            dest.extend_from_slice(if *increment { b"incr " } else { b"decr " });
            dest.extend_from_slice(key);
            dest.push(b' ');
            write_decimal_u64(dest, *delta);
            write_noreply(dest, *noreply);
            dest.extend_from_slice(b"\r\n");
        }
        MemcachedRequest::Touch {
            key,
            exptime,
            noreply,
        } => {
            dest.extend_from_slice(b"touch ");
            dest.extend_from_slice(key);
            dest.push(b' ');
            write_decimal_u32(dest, *exptime);
            write_noreply(dest, *noreply);
            dest.extend_from_slice(b"\r\n");
        }
        MemcachedRequest::FlushAll { delay, noreply } => {
            dest.extend_from_slice(b"flush_all");
            if let Some(delay) = delay {
                dest.push(b' ');
                write_decimal_u32(dest, *delay);
            }
            write_noreply(dest, *noreply);
            dest.extend_from_slice(b"\r\n");
        }
        MemcachedRequest::Stats { args } => {
            dest.extend_from_slice(b"stats");
            if !args.is_empty() {
                dest.push(b' ');
                dest.extend_from_slice(args);
            }
            dest.extend_from_slice(b"\r\n");
        }
        MemcachedRequest::Version => dest.extend_from_slice(b"version\r\n"),
        MemcachedRequest::Quit => dest.extend_from_slice(b"quit\r\n"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::memcached::parse_command;

    #[test]
    fn from_command_keeps_the_multi_get_keys_span_untouched() {
        let raw = Bytes::from_static(b"get a b c\r\n");
        let (command, _) = parse_command(&raw).unwrap();
        let request = MemcachedRequest::from_command(&raw, &command);
        assert_eq!(
            request,
            MemcachedRequest::Get {
                keys: Bytes::from_static(b"a b c"),
                gets: false,
            }
        );
    }

    #[test]
    fn iter_keys_splits_the_span_lazily() {
        let keys = Bytes::from_static(b"a b c");
        let collected: Vec<Bytes> = iter_keys(&keys).collect();
        assert_eq!(
            collected,
            vec![
                Bytes::from_static(b"a"),
                Bytes::from_static(b"b"),
                Bytes::from_static(b"c"),
            ]
        );
    }

    #[test]
    fn iter_keys_skips_runs_of_consecutive_spaces() {
        let keys = Bytes::from_static(b" a  b ");
        let collected: Vec<Bytes> = iter_keys(&keys).collect();
        assert_eq!(collected, vec![Bytes::from_static(b"a"), Bytes::from_static(b"b")]);
    }

    #[test]
    fn iter_keys_over_a_single_key_yields_exactly_one_item() {
        let keys = Bytes::from_static(b"mykey");
        let collected: Vec<Bytes> = iter_keys(&keys).collect();
        assert_eq!(collected, vec![Bytes::from_static(b"mykey")]);
    }

    #[test]
    fn iter_keys_over_an_empty_span_yields_nothing() {
        let keys = Bytes::new();
        assert_eq!(iter_keys(&keys).count(), 0);
    }

    #[test]
    fn from_command_preserves_store_fields() {
        let raw = Bytes::from_static(b"set k 5 60 5\r\nhello\r\n");
        let (command, _) = parse_command(&raw).unwrap();
        let request = MemcachedRequest::from_command(&raw, &command);
        assert_eq!(
            request,
            MemcachedRequest::Store {
                mode: StoreMode::Set,
                key: Bytes::from_static(b"k"),
                flags: 5,
                exptime: 60,
                value: Bytes::from_static(b"hello"),
                noreply: false,
            }
        );
    }

    #[test]
    fn is_noreply_true_only_for_mutation_family() {
        assert!(
            MemcachedRequest::Delete {
                key: Bytes::from_static(b"k"),
                noreply: true,
            }
            .is_noreply()
        );
        assert!(
            !MemcachedRequest::Get {
                keys: Bytes::from_static(b"k"),
                gets: false,
            }
            .is_noreply()
        );
    }

    #[test]
    fn encode_request_round_trips_through_parse_command() {
        let request = MemcachedRequest::Store {
            mode: StoreMode::Set,
            key: Bytes::from_static(b"mykey"),
            flags: 3,
            exptime: 60,
            value: Bytes::from_static(b"payload"),
            noreply: false,
        };
        let mut wire = Vec::new();
        encode_request(&request, &mut wire);
        let raw = Bytes::from(wire);
        let (command, used) = parse_command(&raw).unwrap();
        assert_eq!(used, raw.len());
        assert_eq!(MemcachedRequest::from_command(&raw, &command), request);
    }

    #[test]
    fn encode_request_multi_get_round_trips() {
        let request = MemcachedRequest::Get {
            keys: Bytes::from_static(b"a b"),
            gets: true,
        };
        let mut wire = Vec::new();
        encode_request(&request, &mut wire);
        assert_eq!(wire, b"gets a b\r\n");
    }

    #[test]
    fn encode_request_noreply_delete_round_trips() {
        let request = MemcachedRequest::Delete {
            key: Bytes::from_static(b"k"),
            noreply: true,
        };
        let mut wire = Vec::new();
        encode_request(&request, &mut wire);
        assert_eq!(wire, b"delete k noreply\r\n");
    }
}
