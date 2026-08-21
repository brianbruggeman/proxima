//! Host-side decode of one proxima-vm hypercall exit into a borrowed view
//! over the payload it named — the counterpart to the guest-side trap in
//! `guests/lambda/src/hypercall.rs`. `hvc #0` (aarch64) and `out dx, al`
//! (x86_64) both place a verb and a `(pointer, length)` pair in registers
//! before trapping to the host (`guests/lambda/src/hypercall.rs:33-47,62-78`);
//! this module turns those three already-recovered values plus a borrowed
//! guest-memory slice into a typed view, with no I/O of its own.
//!
//! Mirrors the borrowed-view codec shape every other proxima wire parser
//! uses ([`crate::elf::parse_elf`] in this crate; `proxima_protocols::nvme::command`;
//! `proxima_protocols::quic::packet::header`): a free function over `&[u8]`,
//! never a `Pipe` — deciding whether a hypercall's payload pointer lands
//! inside guest memory is a byte-range check, not a stream to transform.
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). No allocation, no syscall, no `Pipe`.
//!
//! # What this module does not do
//!
//! It does not read vCPU registers — recovering `verb`/`pointer`/`length`
//! from `HV_EXIT_REASON_EXCEPTION` (`src/backend_macos.c:118-137`) or
//! `KVM_EXIT_IO` (`src/backend_linux.c:154-164`) is a tier-2 driver leaf
//! that does not exist yet (`tools/proxima-vm/ROADMAP.md` M1's dispatch
//! component). It does not decode the payload bytes into a `ChildRequest` —
//! that is the dispatch layer, which owns postcard decoding and routing by
//! verb. It does not validate `verb` against a known set — an unrecognized
//! verb is a routing decision, not an ABI-decode failure.

/// Why [`decode_hypercall`] rejected a hypercall's payload pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AbiError {
    /// `pointer + length` either overflowed `u64` or reached past the end
    /// of the guest-memory slice the caller supplied.
    PayloadOutOfRange {
        pointer: u64,
        length: u64,
        memory_len: usize,
    },
}

impl core::fmt::Display for AbiError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PayloadOutOfRange {
                pointer,
                length,
                memory_len,
            } => write!(
                formatter,
                "hypercall payload pointer {pointer:#x} length {length} exceeds guest memory length {memory_len}"
            ),
        }
    }
}

impl core::error::Error for AbiError {}

/// Borrowed view over one decoded hypercall: the verb the guest placed in
/// `x0` (aarch64) / `dx` (x86_64), and the payload bytes at
/// `[pointer, pointer + length)` in the guest-memory slice the caller
/// supplied to [`decode_hypercall`].
#[derive(Debug, Clone, Copy)]
pub struct HypercallView<'a> {
    verb: u16,
    payload: &'a [u8],
}

impl<'a> HypercallView<'a> {
    /// The verb the guest placed in `x0` (aarch64) / `dx` (x86_64) —
    /// `guests/lambda/src/main.rs`'s `CHILD_REQUEST_READ_VERB` reuses the
    /// postcard variant discriminant so the host can route without
    /// decoding postcard first.
    #[must_use]
    pub fn verb(&self) -> u16 {
        self.verb
    }

    /// The payload bytes the hypercall named, borrowed from the guest
    /// memory the caller passed to [`decode_hypercall`].
    #[must_use]
    pub fn payload(&self) -> &'a [u8] {
        self.payload
    }
}

/// Decode one hypercall exit's already-recovered register values —
/// `verb`, `pointer`, `length` — against a borrowed `guest_memory` slice,
/// producing a [`HypercallView`] whose `payload()` borrows the bytes at
/// `[pointer, pointer + length)`.
///
/// `pointer` and `length` come straight from the guest's `x1`/`x2`
/// (aarch64) or `rdi`/`rsi` (x86_64) at the moment of the trap
/// (`guests/lambda/src/hypercall.rs:33-47,62-78`). This function performs
/// no register read itself; the caller — a tier-2 driver leaf — is
/// responsible for recovering the three values from the vCPU exit and for
/// ensuring `guest_memory` is the region the guest's pointer is relative
/// to.
///
/// # Errors
///
/// Returns [`AbiError::PayloadOutOfRange`] when `pointer + length`
/// overflows `u64` or reaches past the end of `guest_memory`.
pub fn decode_hypercall(
    verb: u16,
    pointer: u64,
    length: u64,
    guest_memory: &[u8],
) -> Result<HypercallView<'_>, AbiError> {
    let end = pointer
        .checked_add(length)
        .filter(|end| *end <= guest_memory.len() as u64)
        .ok_or(AbiError::PayloadOutOfRange {
            pointer,
            length,
            memory_len: guest_memory.len(),
        })?;
    let payload = &guest_memory[pointer as usize..end as usize];
    Ok(HypercallView { verb, payload })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// `ChildRequest::Read { path: "/etc/hostname", max_bytes: 256, offset: 0 }`,
    /// postcard-encoded, byte-for-byte the buffer `src/dispatch.rs:124-138`'s
    /// `wire_format_round_trips_for_parity` pins as `expected` — the same
    /// pinned bytes `guests/lambda/src/main.rs`'s
    /// `CHILD_REQUEST_READ_WIRE_BYTES` carries onto the guest side of this
    /// same channel.
    const CHILD_REQUEST_READ_WIRE_BYTES: [u8; 18] = [
        0x00, 13, b'/', b'e', b't', b'c', b'/', b'h', b'o', b's', b't', b'n', b'a', b'm', b'e',
        0x80, 0x02, 0x00,
    ];

    /// Postcard variant discriminant for `ChildRequest::Read`, reused as the
    /// hypercall verb — matches `guests/lambda/src/main.rs`'s
    /// `CHILD_REQUEST_READ_VERB`.
    const CHILD_REQUEST_READ_VERB: u16 = 0x00;

    /// A guest-memory buffer with the pinned request bytes sitting at a
    /// nonzero offset, so tests exercise the pointer arithmetic instead of
    /// happening to work only when `pointer == 0`.
    fn guest_memory_with_pinned_request_at(offset: usize) -> [u8; 32] {
        let mut memory = [0xaa_u8; 32];
        memory[offset..offset + CHILD_REQUEST_READ_WIRE_BYTES.len()]
            .copy_from_slice(&CHILD_REQUEST_READ_WIRE_BYTES);
        memory
    }

    #[test]
    fn decodes_the_pinned_child_request_read_payload() {
        let memory = guest_memory_with_pinned_request_at(4);

        let view = decode_hypercall(
            CHILD_REQUEST_READ_VERB,
            4,
            CHILD_REQUEST_READ_WIRE_BYTES.len() as u64,
            &memory,
        )
        .expect("pointer and length are within guest memory");

        assert_eq!(view.verb(), CHILD_REQUEST_READ_VERB);
        assert_eq!(view.payload(), &CHILD_REQUEST_READ_WIRE_BYTES);
    }

    #[test]
    fn decodes_a_zero_length_payload_at_the_end_of_memory() {
        let memory = [0_u8; 8];

        let view = decode_hypercall(1, 8, 0, &memory).expect("zero-length payload at the boundary is in range");

        assert_eq!(view.verb(), 1);
        assert!(view.payload().is_empty());
    }

    #[test]
    fn rejects_a_pointer_length_pair_reaching_past_guest_memory() {
        let memory = guest_memory_with_pinned_request_at(4);
        let out_of_range_length = (memory.len() - 4 + 1) as u64;

        let error = decode_hypercall(CHILD_REQUEST_READ_VERB, 4, out_of_range_length, &memory).unwrap_err();

        assert_eq!(
            error,
            AbiError::PayloadOutOfRange {
                pointer: 4,
                length: out_of_range_length,
                memory_len: memory.len(),
            }
        );
    }

    #[test]
    fn rejects_a_pointer_beyond_guest_memory_even_with_zero_length() {
        let memory = [0_u8; 8];

        let error = decode_hypercall(0, 9, 0, &memory).unwrap_err();

        assert_eq!(
            error,
            AbiError::PayloadOutOfRange {
                pointer: 9,
                length: 0,
                memory_len: memory.len(),
            }
        );
    }

    #[test]
    fn rejects_a_pointer_plus_length_that_overflows_u64() {
        let memory = [0_u8; 8];

        let error = decode_hypercall(0, u64::MAX, 1, &memory).unwrap_err();

        assert_eq!(
            error,
            AbiError::PayloadOutOfRange {
                pointer: u64::MAX,
                length: 1,
                memory_len: memory.len(),
            }
        );
    }

    #[test]
    fn abi_error_display_names_the_offending_pointer() {
        let error = AbiError::PayloadOutOfRange {
            pointer: 0x100,
            length: 32,
            memory_len: 64,
        };

        assert_eq!(
            error.to_string(),
            "hypercall payload pointer 0x100 length 32 exceeds guest memory length 64"
        );
    }
}
