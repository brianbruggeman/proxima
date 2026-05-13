use super::error::DecodeError;
use super::raw::{read_u16, read_u32, read_u64, write_u16, write_u32, write_u64};

/// Fixed size of an NVMe Submission Queue Entry. Every command the host posts to
/// a submission-queue ring occupies exactly this many bytes.
pub const ENTRY_LEN: usize = 64;

/// The data-transfer direction encoded in the low two bits of the opcode
/// (NVMe base spec, Command Dword 0 — Opcode field, bits 01:00). The controller
/// uses this to know which way the data pointer moves before it has parsed the
/// rest of the command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataTransfer {
    /// No data transfer (e.g. Flush).
    None,
    /// Host to controller (a write).
    HostToController,
    /// Controller to host (a read).
    ControllerToHost,
    /// Bidirectional.
    Bidirectional,
}

impl DataTransfer {
    #[must_use]
    fn from_opcode(opcode: u8) -> Self {
        match opcode & 0b11 {
            0b00 => Self::None,
            0b01 => Self::HostToController,
            0b10 => Self::ControllerToHost,
            _ => Self::Bidirectional,
        }
    }
}

/// Borrowed view over a 64-byte Submission Queue Entry sitting in the host's
/// queue memory. All multi-byte fields are little-endian (host memory order, not
/// network order) — this is host/controller shared DRAM, not a wire.
#[derive(Debug, Clone, Copy)]
pub struct SubmissionEntry<'sqe> {
    bytes: &'sqe [u8],
}

impl<'sqe> SubmissionEntry<'sqe> {
    pub fn parse(bytes: &'sqe [u8]) -> Result<Self, DecodeError> {
        if bytes.len() < ENTRY_LEN {
            return Err(DecodeError::Truncated {
                need: ENTRY_LEN,
                got: bytes.len(),
            });
        }
        Ok(Self { bytes })
    }

    #[must_use]
    pub fn opcode(&self) -> u8 {
        self.bytes[0]
    }

    #[must_use]
    pub fn data_transfer(&self) -> DataTransfer {
        DataTransfer::from_opcode(self.opcode())
    }

    /// Fused-operation field (CDW0 bits 09:08): 00 normal, 01 first half, 10
    /// second half.
    #[must_use]
    pub fn fuse(&self) -> u8 {
        self.bytes[1] & 0b11
    }

    /// PRP-or-SGL-for-data-transfer selector (CDW0 bits 15:14).
    #[must_use]
    pub fn psdt(&self) -> u8 {
        (self.bytes[1] >> 6) & 0b11
    }

    #[must_use]
    pub fn command_id(&self) -> u16 {
        read_u16(self.bytes, 2)
    }

    #[must_use]
    pub fn namespace_id(&self) -> u32 {
        read_u32(self.bytes, 4)
    }

    #[must_use]
    pub fn metadata_ptr(&self) -> u64 {
        read_u64(self.bytes, 16)
    }

    /// First data pointer (PRP1, or the low 8 bytes of an inline SGL).
    #[must_use]
    pub fn data_ptr1(&self) -> u64 {
        read_u64(self.bytes, 24)
    }

    /// Second data pointer (PRP2, or the high 8 bytes of an inline SGL).
    #[must_use]
    pub fn data_ptr2(&self) -> u64 {
        read_u64(self.bytes, 32)
    }

    /// Command-specific dwords 10..=15. `index` is 0-based over that window, so
    /// `command_dword(0)` is CDW10. Indices past 5 read as zero rather than
    /// panicking, keeping the accessor total.
    #[must_use]
    pub fn command_dword(&self, index: usize) -> u32 {
        if index > 5 {
            return 0;
        }
        read_u32(self.bytes, 40 + index * 4)
    }
}

/// Build a 64-byte Submission Queue Entry into `out`, writing every field in
/// little-endian order. Reserved dwords (CDW2/CDW3) are zeroed. Returns the
/// number of bytes written so the caller can advance a ring tail.
#[derive(Debug, Clone, Copy)]
pub struct CommandBuilder {
    opcode: u8,
    flags: u8,
    command_id: u16,
    namespace_id: u32,
    metadata_ptr: u64,
    data_ptr1: u64,
    data_ptr2: u64,
    command_dwords: [u32; 6],
}

impl CommandBuilder {
    #[must_use]
    pub fn new(opcode: u8, command_id: u16) -> Self {
        Self {
            opcode,
            flags: 0,
            command_id,
            namespace_id: 0,
            metadata_ptr: 0,
            data_ptr1: 0,
            data_ptr2: 0,
            command_dwords: [0; 6],
        }
    }

    /// The command identifier this builder will stamp into the SQE — the key a
    /// reaper matches the returning completion against.
    #[must_use]
    pub fn command_id(&self) -> u16 {
        self.command_id
    }

    /// The opcode this builder will write at byte 0.
    #[must_use]
    pub fn opcode(&self) -> u8 {
        self.opcode
    }

    #[must_use]
    pub fn namespace_id(mut self, namespace_id: u32) -> Self {
        self.namespace_id = namespace_id;
        self
    }

    #[must_use]
    pub fn data_ptrs(mut self, prp1: u64, prp2: u64) -> Self {
        self.data_ptr1 = prp1;
        self.data_ptr2 = prp2;
        self
    }

    #[must_use]
    pub fn metadata_ptr(mut self, metadata_ptr: u64) -> Self {
        self.metadata_ptr = metadata_ptr;
        self
    }

    /// Set command-specific dword `index` (0 == CDW10). Out-of-range indices are
    /// ignored so the builder chain stays infallible.
    #[must_use]
    pub fn command_dword(mut self, index: usize, value: u32) -> Self {
        if let Some(slot) = self.command_dwords.get_mut(index) {
            *slot = value;
        }
        self
    }

    pub fn write(&self, out: &mut [u8]) -> Result<usize, DecodeError> {
        if out.len() < ENTRY_LEN {
            return Err(DecodeError::Truncated {
                need: ENTRY_LEN,
                got: out.len(),
            });
        }
        let entry = &mut out[..ENTRY_LEN];
        entry[0] = self.opcode;
        entry[1] = self.flags;
        write_u16(entry, 2, self.command_id);
        write_u32(entry, 4, self.namespace_id);
        write_u32(entry, 8, 0);
        write_u32(entry, 12, 0);
        write_u64(entry, 16, self.metadata_ptr);
        write_u64(entry, 24, self.data_ptr1);
        write_u64(entry, 32, self.data_ptr2);
        for (index, value) in self.command_dwords.iter().enumerate() {
            write_u32(entry, 40 + index * 4, *value);
        }
        Ok(ENTRY_LEN)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proptest::prelude::*;

    // NVM Read (opcode 0x02): controller-to-host transfer of NLB+1 logical
    // blocks starting at SLBA. CDW10/CDW11 carry SLBA low/high, CDW12 the
    // 0-based block count. This is the canonical hot-path I/O command.
    const OPC_READ: u8 = 0x02;

    #[test]
    fn read_command_lands_at_spec_byte_offsets() {
        let mut sqe = [0u8; ENTRY_LEN];
        let slba: u64 = 0x0000_0000_1234_5678;
        let written = CommandBuilder::new(OPC_READ, 0x0007)
            .namespace_id(1)
            .data_ptrs(0xdead_beef_0000_1000, 0)
            .command_dword(0, slba as u32)
            .command_dword(1, (slba >> 32) as u32)
            .command_dword(2, 7)
            .write(&mut sqe)
            .expect("64-byte buffer fits an SQE");
        assert_eq!(written, ENTRY_LEN);

        // opcode at byte 0, CID little-endian at bytes 2..4, NSID at 4..8.
        assert_eq!(sqe[0], OPC_READ);
        assert_eq!(&sqe[2..4], &0x0007u16.to_le_bytes());
        assert_eq!(&sqe[4..8], &1u32.to_le_bytes());
        // PRP1 at bytes 24..32, little-endian.
        assert_eq!(&sqe[24..32], &0xdead_beef_0000_1000u64.to_le_bytes());
        // CDW10 (SLBA low) at bytes 40..44.
        assert_eq!(&sqe[40..44], &(slba as u32).to_le_bytes());
    }

    #[test]
    fn build_then_parse_round_trips_fields() {
        let mut sqe = [0u8; ENTRY_LEN];
        CommandBuilder::new(OPC_READ, 0x00ab)
            .namespace_id(0x0000_0042)
            .metadata_ptr(0x1111_2222_3333_4444)
            .data_ptrs(0xaaaa_bbbb_cccc_dddd, 0x9999_8888_7777_6666)
            .command_dword(5, 0xfeed_face)
            .write(&mut sqe)
            .unwrap();

        let view = SubmissionEntry::parse(&sqe).unwrap();
        assert_eq!(view.opcode(), OPC_READ);
        assert_eq!(view.data_transfer(), DataTransfer::ControllerToHost);
        assert_eq!(view.command_id(), 0x00ab);
        assert_eq!(view.namespace_id(), 0x0000_0042);
        assert_eq!(view.metadata_ptr(), 0x1111_2222_3333_4444);
        assert_eq!(view.data_ptr1(), 0xaaaa_bbbb_cccc_dddd);
        assert_eq!(view.data_ptr2(), 0x9999_8888_7777_6666);
        assert_eq!(view.command_dword(5), 0xfeed_face);
    }

    #[test]
    fn data_transfer_decodes_from_opcode_low_bits() {
        // Flush (0x00) -> none, Write (0x01) -> host-to-controller,
        // Read (0x02) -> controller-to-host, 0x03 -> bidirectional.
        for (opcode, expected) in [
            (0x00u8, DataTransfer::None),
            (0x01, DataTransfer::HostToController),
            (0x02, DataTransfer::ControllerToHost),
            (0x03, DataTransfer::Bidirectional),
        ] {
            let mut sqe = [0u8; ENTRY_LEN];
            CommandBuilder::new(opcode, 0).write(&mut sqe).unwrap();
            assert_eq!(
                SubmissionEntry::parse(&sqe).unwrap().data_transfer(),
                expected
            );
        }
    }

    #[test]
    fn out_of_range_command_dword_reads_zero_not_panic() {
        let sqe = [0xffu8; ENTRY_LEN];
        let view = SubmissionEntry::parse(&sqe).unwrap();
        assert_eq!(view.command_dword(6), 0);
        assert_eq!(view.command_dword(usize::MAX), 0);
    }

    #[test]
    fn short_buffer_is_truncated_on_parse_and_write() {
        assert_eq!(
            SubmissionEntry::parse(&[0u8; ENTRY_LEN - 1]).unwrap_err(),
            DecodeError::Truncated {
                need: ENTRY_LEN,
                got: ENTRY_LEN - 1
            }
        );
        let mut tiny = [0u8; ENTRY_LEN - 1];
        assert_eq!(
            CommandBuilder::new(OPC_READ, 1)
                .write(&mut tiny)
                .unwrap_err(),
            DecodeError::Truncated {
                need: ENTRY_LEN,
                got: ENTRY_LEN - 1
            }
        );
    }

    proptest! {
        #[test]
        fn parse_never_panics_on_arbitrary_bytes(
            data in prop::collection::vec(any::<u8>(), 0..256),
        ) {
            if let Ok(view) = SubmissionEntry::parse(&data) {
                // every accessor is total over a 64-byte slot
                let _ = view.opcode();
                let _ = view.command_id();
                let _ = view.namespace_id();
                let _ = view.data_ptr1();
                for index in 0..8 {
                    let _ = view.command_dword(index);
                }
            }
        }

        #[test]
        fn build_then_parse_round_trips_arbitrary_fields(
            opcode in any::<u8>(),
            command_id in any::<u16>(),
            namespace_id in any::<u32>(),
            prp1 in any::<u64>(),
            prp2 in any::<u64>(),
            dwords in prop::array::uniform6(any::<u32>()),
        ) {
            let mut sqe = [0u8; ENTRY_LEN];
            let mut builder = CommandBuilder::new(opcode, command_id)
                .namespace_id(namespace_id)
                .data_ptrs(prp1, prp2);
            for (index, value) in dwords.iter().enumerate() {
                builder = builder.command_dword(index, *value);
            }
            builder.write(&mut sqe).unwrap();

            let view = SubmissionEntry::parse(&sqe).unwrap();
            prop_assert_eq!(view.opcode(), opcode);
            prop_assert_eq!(view.command_id(), command_id);
            prop_assert_eq!(view.namespace_id(), namespace_id);
            prop_assert_eq!(view.data_ptr1(), prp1);
            prop_assert_eq!(view.data_ptr2(), prp2);
            for (index, value) in dwords.iter().enumerate() {
                prop_assert_eq!(view.command_dword(index), *value);
            }
        }
    }
}
