use proxima_protocols::nvme::CommandBuilder;
use thiserror::Error;

const READ_OPCODE: u8 = 0x02;
const MAX_BLOCKS: u64 = 65_536;

/// A byte-aligned NVMe read request for a caller-owned DMA destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadRequest {
    command_id: u16,
    namespace_id: u32,
    byte_offset: u64,
    byte_length: u64,
    logical_block_size: u32,
    prp1: u64,
    prp2: u64,
}

impl ReadRequest {
    /// Build a read request. The destination addresses are PRP1 and PRP2;
    /// SPDK remains responsible for allocating DMA-safe memory and supplying
    /// any PRP list required by a larger transfer.
    pub fn new(
        command_id: u16,
        namespace_id: u32,
        byte_offset: u64,
        byte_length: u64,
        logical_block_size: u32,
        prp1: u64,
        prp2: u64,
    ) -> Result<Self, ReadRequestError> {
        if logical_block_size == 0 {
            return Err(ReadRequestError::ZeroBlockSize);
        }
        let block_size = u64::from(logical_block_size);
        if !byte_offset.is_multiple_of(block_size) {
            return Err(ReadRequestError::UnalignedOffset { byte_offset });
        }
        if byte_length == 0 || !byte_length.is_multiple_of(block_size) {
            return Err(ReadRequestError::UnalignedLength { byte_length });
        }
        let block_count = byte_length / block_size;
        if block_count > MAX_BLOCKS {
            return Err(ReadRequestError::TooManyBlocks { block_count });
        }
        Ok(Self {
            command_id,
            namespace_id,
            byte_offset,
            byte_length,
            logical_block_size,
            prp1,
            prp2,
        })
    }

    /// Encode this request as an NVMe NVM Read command.
    #[must_use]
    pub fn command(&self) -> CommandBuilder {
        let block_size = u64::from(self.logical_block_size);
        let slba = self.byte_offset / block_size;
        let block_count = self.byte_length / block_size;
        CommandBuilder::new(READ_OPCODE, self.command_id)
            .namespace_id(self.namespace_id)
            .data_ptrs(self.prp1, self.prp2)
            .command_dword(0, slba as u32)
            .command_dword(1, (slba >> 32) as u32)
            .command_dword(2, (block_count - 1) as u32)
    }
}

/// Why an NVMe read request cannot be encoded safely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ReadRequestError {
    #[error("logical block size must be nonzero")]
    ZeroBlockSize,
    #[error("read offset {byte_offset} is not logical-block aligned")]
    UnalignedOffset { byte_offset: u64 },
    #[error("read length {byte_length} is zero or not logical-block aligned")]
    UnalignedLength { byte_length: u64 },
    #[error("read has {block_count} blocks; NVMe permits at most 65536")]
    TooManyBlocks { block_count: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use proxima_protocols::nvme::SubmissionEntry;

    #[test]
    fn read_request_matches_nvme_wire_fields() {
        let request = ReadRequest::new(7, 1, 0x1234_5600, 4096, 512, 0x1000, 0x2000)
            .expect("aligned read request");
        let mut bytes = [0u8; 64];
        request.command().write(&mut bytes).expect("encode read command");
        let view = SubmissionEntry::parse(&bytes).expect("parse read command");

        assert_eq!(view.opcode(), READ_OPCODE);
        assert_eq!(view.command_id(), 7);
        assert_eq!(view.namespace_id(), 1);
        assert_eq!(view.data_ptr1(), 0x1000);
        assert_eq!(view.data_ptr2(), 0x2000);
        assert_eq!(view.command_dword(0), 0x1234_5600 / 512);
        assert_eq!(view.command_dword(2), 7);
    }

    #[test]
    fn read_request_rejects_unaligned_ranges_and_oversized_transfers() {
        assert_eq!(
            ReadRequest::new(1, 1, 1, 512, 512, 0, 0),
            Err(ReadRequestError::UnalignedOffset { byte_offset: 1 })
        );
        assert_eq!(
            ReadRequest::new(1, 1, 0, 513, 512, 0, 0),
            Err(ReadRequestError::UnalignedLength { byte_length: 513 })
        );
        assert_eq!(
            ReadRequest::new(1, 1, 0, 65_537 * 512, 512, 0, 0),
            Err(ReadRequestError::TooManyBlocks { block_count: 65_537 })
        );
    }
}
