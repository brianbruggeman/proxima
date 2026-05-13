use super::error::DecodeError;
use super::raw::{read_u16, read_u32, write_u16, write_u32};

/// Fixed size of an NVMe Completion Queue Entry. The controller posts exactly
/// this many bytes per completed command.
pub const ENTRY_LEN: usize = 16;

/// The 16-bit status field (CQE dword 3, bits 31:16) decoded into its parts.
///
/// The phase bit is the load-bearing field for lock-free polling: the consumer
/// flips an expected-phase bit every time the completion ring wraps, so a slot
/// whose phase matches the expectation is a fresh completion and one that does
/// not is stale leftover memory. No interrupt or shared counter needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatusField(u16);

impl StatusField {
    #[must_use]
    pub fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[must_use]
    pub fn bits(self) -> u16 {
        self.0
    }

    /// Phase Tag (P), bit 0.
    #[must_use]
    pub fn phase(self) -> bool {
        self.0 & 0b1 != 0
    }

    /// Status Code (SC), bits 08:01.
    #[must_use]
    pub fn status_code(self) -> u8 {
        ((self.0 >> 1) & 0xff) as u8
    }

    /// Status Code Type (SCT), bits 11:09.
    #[must_use]
    pub fn status_code_type(self) -> u8 {
        ((self.0 >> 9) & 0b111) as u8
    }

    /// Command Retry Delay (CRD), bits 13:12.
    #[must_use]
    pub fn retry_delay(self) -> u8 {
        ((self.0 >> 12) & 0b11) as u8
    }

    /// More (M), bit 14 — extra status info available via Get Log Page.
    #[must_use]
    pub fn more(self) -> bool {
        self.0 & (1 << 14) != 0
    }

    /// Do Not Retry (DNR), bit 15.
    #[must_use]
    pub fn do_not_retry(self) -> bool {
        self.0 & (1 << 15) != 0
    }

    /// Success is SCT == 0 (Generic) and SC == 0 (Successful Completion). The
    /// phase bit is excluded — it is a ring-protocol marker, not a verdict.
    #[must_use]
    pub fn is_success(self) -> bool {
        self.status_code() == 0 && self.status_code_type() == 0
    }
}

/// Borrowed view over a 16-byte Completion Queue Entry in the host's queue
/// memory. Little-endian, like the submission side.
#[derive(Debug, Clone, Copy)]
pub struct CompletionEntry<'cqe> {
    bytes: &'cqe [u8],
}

impl<'cqe> CompletionEntry<'cqe> {
    pub fn parse(bytes: &'cqe [u8]) -> Result<Self, DecodeError> {
        if bytes.len() < ENTRY_LEN {
            return Err(DecodeError::Truncated {
                need: ENTRY_LEN,
                got: bytes.len(),
            });
        }
        Ok(Self { bytes })
    }

    /// Command-specific dword 0 — the result value for commands that return one.
    #[must_use]
    pub fn command_specific(&self) -> u32 {
        read_u32(self.bytes, 0)
    }

    /// Submission Queue Head Pointer (SQHD): how far the controller has consumed
    /// the submission ring, so the host can reclaim those SQE slots.
    #[must_use]
    pub fn sq_head(&self) -> u16 {
        read_u16(self.bytes, 8)
    }

    /// Submission Queue Identifier (SQID) this completion is for.
    #[must_use]
    pub fn sq_id(&self) -> u16 {
        read_u16(self.bytes, 10)
    }

    /// Command Identifier (CID) echoed back from the originating SQE.
    #[must_use]
    pub fn command_id(&self) -> u16 {
        read_u16(self.bytes, 12)
    }

    #[must_use]
    pub fn status(&self) -> StatusField {
        StatusField::from_bits(read_u16(self.bytes, 14))
    }

    /// Command Identifier and status field together, read as the single dword 3
    /// the controller writes them in (CID in the low 16 bits, status in the
    /// high 16). The per-completion reap wants both at once — folding them into
    /// one aligned u32 load matches what a struct-cast driver does and keeps the
    /// hot poll loop vectorisable. The phase tag lives in `status.phase()`.
    #[must_use]
    pub fn command_id_and_status(&self) -> (u16, StatusField) {
        let dword = read_u32(self.bytes, 12);
        (
            (dword & 0xffff) as u16,
            StatusField::from_bits((dword >> 16) as u16),
        )
    }

    /// The phase tag of this slot, read directly without decoding the rest of
    /// the status field — the inner-loop poll only needs this bit.
    #[must_use]
    pub fn phase(&self) -> bool {
        self.bytes[14] & 0b1 != 0
    }
}

/// Build a 16-byte Completion Queue Entry into `out` (controller-side, or a test
/// double standing in for the controller). Returns bytes written.
pub fn write_completion(
    out: &mut [u8],
    command_specific: u32,
    sq_head: u16,
    sq_id: u16,
    command_id: u16,
    status: StatusField,
) -> Result<usize, DecodeError> {
    if out.len() < ENTRY_LEN {
        return Err(DecodeError::Truncated {
            need: ENTRY_LEN,
            got: out.len(),
        });
    }
    let entry = &mut out[..ENTRY_LEN];
    write_u32(entry, 0, command_specific);
    write_u32(entry, 4, 0);
    write_u16(entry, 8, sq_head);
    write_u16(entry, 10, sq_id);
    write_u16(entry, 12, command_id);
    write_u16(entry, 14, status.bits());
    Ok(ENTRY_LEN)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn success_status_decodes_to_all_clear() {
        // phase=1, SC=0, SCT=0 -> the wire status word is 0x0001.
        let status = StatusField::from_bits(0x0001);
        assert!(status.phase());
        assert_eq!(status.status_code(), 0);
        assert_eq!(status.status_code_type(), 0);
        assert!(!status.do_not_retry());
        assert!(status.is_success());
    }

    #[test]
    fn packed_status_field_splits_at_spec_bit_boundaries() {
        // phase=1 | SC=0x81<<1 | SCT=0x02<<9 | DNR=1<<15 == 0x8503.
        let status = StatusField::from_bits(0x8503);
        assert!(status.phase());
        assert_eq!(status.status_code(), 0x81);
        assert_eq!(status.status_code_type(), 0x02);
        assert!(status.do_not_retry());
        assert!(!status.more());
        assert!(!status.is_success());
    }

    #[test]
    fn completion_lands_at_spec_byte_offsets_and_round_trips() {
        let status = StatusField::from_bits(0x8503);
        let mut cqe = [0u8; ENTRY_LEN];
        let written =
            write_completion(&mut cqe, 0x1234_5678, 0x0042, 0x0001, 0x00ab, status).unwrap();
        assert_eq!(written, ENTRY_LEN);

        // SQHD at 8..10, SQID at 10..12, CID at 12..14, status at 14..16.
        assert_eq!(&cqe[8..10], &0x0042u16.to_le_bytes());
        assert_eq!(&cqe[12..14], &0x00abu16.to_le_bytes());
        assert_eq!(&cqe[14..16], &0x8503u16.to_le_bytes());

        let view = CompletionEntry::parse(&cqe).unwrap();
        assert_eq!(view.command_specific(), 0x1234_5678);
        assert_eq!(view.sq_head(), 0x0042);
        assert_eq!(view.sq_id(), 0x0001);
        assert_eq!(view.command_id(), 0x00ab);
        assert_eq!(view.status(), status);
        assert!(view.phase());
    }

    #[test]
    fn folded_accessor_matches_individual_reads() {
        let status = StatusField::from_bits(0x8503);
        let mut cqe = [0u8; ENTRY_LEN];
        write_completion(&mut cqe, 0, 0, 0, 0x00ab, status).unwrap();
        let view = CompletionEntry::parse(&cqe).unwrap();
        let (cid, folded_status) = view.command_id_and_status();
        assert_eq!(cid, view.command_id());
        assert_eq!(folded_status, view.status());
    }

    #[test]
    fn phase_accessor_matches_full_status_phase() {
        for bits in [0x0000u16, 0x0001, 0xfffe, 0xffff] {
            let mut cqe = [0u8; ENTRY_LEN];
            write_completion(&mut cqe, 0, 0, 0, 0, StatusField::from_bits(bits)).unwrap();
            let view = CompletionEntry::parse(&cqe).unwrap();
            assert_eq!(view.phase(), view.status().phase());
        }
    }

    #[test]
    fn short_buffer_is_truncated() {
        assert_eq!(
            CompletionEntry::parse(&[0u8; ENTRY_LEN - 1]).unwrap_err(),
            DecodeError::Truncated {
                need: ENTRY_LEN,
                got: ENTRY_LEN - 1
            }
        );
        let mut tiny = [0u8; ENTRY_LEN - 1];
        assert_eq!(
            write_completion(&mut tiny, 0, 0, 0, 0, StatusField::from_bits(0)).unwrap_err(),
            DecodeError::Truncated {
                need: ENTRY_LEN,
                got: ENTRY_LEN - 1
            }
        );
    }

    proptest! {
        #[test]
        fn status_field_components_recompose(bits in any::<u16>()) {
            let status = StatusField::from_bits(bits);
            // the six fields tile the 16-bit word with no overlap or gap
            let recomposed = u16::from(status.phase())
                | (u16::from(status.status_code()) << 1)
                | (u16::from(status.status_code_type()) << 9)
                | (u16::from(status.retry_delay()) << 12)
                | (u16::from(status.more()) << 14)
                | (u16::from(status.do_not_retry()) << 15);
            prop_assert_eq!(recomposed, bits);
        }

        #[test]
        fn write_then_parse_round_trips_arbitrary_fields(
            command_specific in any::<u32>(),
            sq_head in any::<u16>(),
            sq_id in any::<u16>(),
            command_id in any::<u16>(),
            status_bits in any::<u16>(),
        ) {
            let status = StatusField::from_bits(status_bits);
            let mut cqe = [0u8; ENTRY_LEN];
            write_completion(&mut cqe, command_specific, sq_head, sq_id, command_id, status)
                .unwrap();
            let view = CompletionEntry::parse(&cqe).unwrap();
            prop_assert_eq!(view.command_specific(), command_specific);
            prop_assert_eq!(view.sq_head(), sq_head);
            prop_assert_eq!(view.sq_id(), sq_id);
            prop_assert_eq!(view.command_id(), command_id);
            prop_assert_eq!(view.status(), status);
        }

        #[test]
        fn parse_never_panics_on_arbitrary_bytes(
            data in prop::collection::vec(any::<u8>(), 0..64),
        ) {
            if let Ok(view) = CompletionEntry::parse(&data) {
                let _ = view.command_specific();
                let _ = view.sq_head();
                let _ = view.status().is_success();
                let _ = view.phase();
            }
        }
    }
}
