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
//! `Vec<Vec<u8>>` arg-bag the handler has to re-parse per verb shape.

use alloc::vec::Vec;

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
/// shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemcachedRequest {
    Get {
        keys: Vec<Vec<u8>>,
        gets: bool,
    },
    Store {
        mode: StoreMode,
        key: Vec<u8>,
        flags: u32,
        exptime: u32,
        value: Vec<u8>,
        noreply: bool,
    },
    Cas {
        key: Vec<u8>,
        flags: u32,
        exptime: u32,
        cas_unique: u64,
        value: Vec<u8>,
        noreply: bool,
    },
    Delete {
        key: Vec<u8>,
        noreply: bool,
    },
    Counter {
        increment: bool,
        key: Vec<u8>,
        delta: u64,
        noreply: bool,
    },
    Touch {
        key: Vec<u8>,
        exptime: u32,
        noreply: bool,
    },
    FlushAll {
        delay: Option<u32>,
        noreply: bool,
    },
    Stats {
        args: Vec<u8>,
    },
    Version,
    Quit,
}

impl MemcachedRequest {
    /// Lift a borrowed, parse-buffer-tied [`Command`] into an owned,
    /// `'static` request — the async-boundary conversion a handler pipe
    /// needs (the borrowed form cannot outlive the connection's read
    /// buffer past the point the driver calls `Connection::consume`).
    #[must_use]
    pub fn from_command(command: &Command<'_>) -> Self {
        // `Command<'_>`'s fields are all Copy-eligible (`&[u8]`/`u32`/`bool`);
        // cloning is a bitwise copy of borrowed slices, not an allocation —
        // matching on the clone (by value) avoids the `&&[u8]` double
        // indirection a reference-pattern match ergonomics would produce.
        match command.clone() {
            Command::Get { keys, gets } => Self::Get {
                keys: split_keys(keys),
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
                key: key.to_vec(),
                flags,
                exptime,
                value: value.to_vec(),
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
                key: key.to_vec(),
                flags,
                exptime,
                cas_unique,
                value: value.to_vec(),
                noreply,
            },
            Command::Delete { key, noreply } => Self::Delete {
                key: key.to_vec(),
                noreply,
            },
            Command::Counter {
                increment,
                key,
                delta,
                noreply,
            } => Self::Counter {
                increment,
                key: key.to_vec(),
                delta,
                noreply,
            },
            Command::Touch {
                key,
                exptime,
                noreply,
            } => Self::Touch {
                key: key.to_vec(),
                exptime,
                noreply,
            },
            Command::FlushAll { delay, noreply } => Self::FlushAll { delay, noreply },
            Command::Stats { args } => Self::Stats {
                args: args.to_vec(),
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

fn split_keys(joined: &[u8]) -> Vec<Vec<u8>> {
    joined
        .split(|&byte| byte == b' ')
        .filter(|slice| !slice.is_empty())
        .map(<[u8]>::to_vec)
        .collect()
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
/// [`Command`] (there is no existing request encoder to reuse): `Command`'s
/// `Get::keys` borrows one already-space-joined wire slice, while
/// [`MemcachedRequest::Get`] holds its keys as separate owned buffers;
/// joining them into a scratch buffer just to hand it to a
/// `Command`-shaped encoder would cost the same allocation this function
/// does directly, with an extra indirection.
pub fn encode_request(request: &MemcachedRequest, dest: &mut Vec<u8>) {
    match request {
        MemcachedRequest::Get { keys, gets } => {
            dest.extend_from_slice(if *gets { b"gets" } else { b"get" });
            for key in keys {
                dest.push(b' ');
                dest.extend_from_slice(key);
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
    fn from_command_splits_multi_get_keys() {
        let (command, _) = parse_command(b"get a b c\r\n").unwrap();
        let request = MemcachedRequest::from_command(&command);
        assert_eq!(
            request,
            MemcachedRequest::Get {
                keys: vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
                gets: false,
            }
        );
    }

    #[test]
    fn from_command_preserves_store_fields() {
        let (command, _) = parse_command(b"set k 5 60 5\r\nhello\r\n").unwrap();
        let request = MemcachedRequest::from_command(&command);
        assert_eq!(
            request,
            MemcachedRequest::Store {
                mode: StoreMode::Set,
                key: b"k".to_vec(),
                flags: 5,
                exptime: 60,
                value: b"hello".to_vec(),
                noreply: false,
            }
        );
    }

    #[test]
    fn is_noreply_true_only_for_mutation_family() {
        assert!(
            MemcachedRequest::Delete {
                key: b"k".to_vec(),
                noreply: true,
            }
            .is_noreply()
        );
        assert!(
            !MemcachedRequest::Get {
                keys: vec![b"k".to_vec()],
                gets: false,
            }
            .is_noreply()
        );
    }

    #[test]
    fn encode_request_round_trips_through_parse_command() {
        let request = MemcachedRequest::Store {
            mode: StoreMode::Set,
            key: b"mykey".to_vec(),
            flags: 3,
            exptime: 60,
            value: b"payload".to_vec(),
            noreply: false,
        };
        let mut wire = Vec::new();
        encode_request(&request, &mut wire);
        let (command, used) = parse_command(&wire).unwrap();
        assert_eq!(used, wire.len());
        assert_eq!(MemcachedRequest::from_command(&command), request);
    }

    #[test]
    fn encode_request_multi_get_round_trips() {
        let request = MemcachedRequest::Get {
            keys: vec![b"a".to_vec(), b"b".to_vec()],
            gets: true,
        };
        let mut wire = Vec::new();
        encode_request(&request, &mut wire);
        assert_eq!(wire, b"gets a b\r\n");
    }

    #[test]
    fn encode_request_noreply_delete_round_trips() {
        let request = MemcachedRequest::Delete {
            key: b"k".to_vec(),
            noreply: true,
        };
        let mut wire = Vec::new();
        encode_request(&request, &mut wire);
        assert_eq!(wire, b"delete k noreply\r\n");
    }
}
