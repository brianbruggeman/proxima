//! Header types of the kv block file and the length of an encoded file.
//!
//! A block file is a header (fixed fields plus one record per layer) followed by the
//! per-layer `k_even`, `k_odd` and `v` planes. These types describe that header; the
//! length function says how many bytes a file with such a header occupies.

use alloc::vec::Vec;

use crate::error::InteropError;

const HEADER_FIXED_BYTES: usize = 48;
const LAYER_RECORD_BYTES: usize = 24;

/// Shape of one layer's rows inside a block file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockFileLayer {
    /// Rows this layer holds in the file, for example `57`.
    pub rows: u32,
    /// Bytes in one row of the even key plane, for example `2048`.
    pub k_even_row_bytes: u32,
    /// Bytes in one row of the odd key plane, for example `2048`.
    pub k_odd_row_bytes: u32,
    /// Bytes in one row of the value plane, for example `1024`.
    pub v_row_bytes: u32,
    /// Sliding window of a ring layer in rows, for example `512`; `0` for a full layer.
    pub ring_window: u32,
    /// Slot count of a ring layer in rows, for example `1024`; `0` for a full layer.
    pub ring_capacity: u32,
}

/// Header of a kv block file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockFileHeader {
    /// Digest of the model descriptor the rows were produced by.
    pub descriptor_digest: [u8; 16],
    /// Content key the file is stored under, for example `0x0102030405060708`.
    pub content_key: u64,
    /// Position of the first row in the sequence, for example `64`.
    pub base_position: u64,
    /// One record per layer, in layer order.
    pub layers: Vec<BlockFileLayer>,
}

/// Total bytes of a file with this header: fixed header, layer table and every plane.
///
/// Saturates at `usize::MAX`: a header whose size does not fit memory can have no
/// planes to match it.
#[must_use]
pub fn encoded_len(header: &BlockFileHeader) -> usize {
    let table = LAYER_RECORD_BYTES.saturating_mul(header.layers.len());
    header
        .layers
        .iter()
        .fold(HEADER_FIXED_BYTES.saturating_add(table), |total, layer| {
            total.saturating_add(plane_bytes(layer))
        })
}

fn plane_bytes(layer: &BlockFileLayer) -> usize {
    let row_bytes = (layer.k_even_row_bytes as usize)
        .saturating_add(layer.k_odd_row_bytes as usize)
        .saturating_add(layer.v_row_bytes as usize);
    (layer.rows as usize).saturating_mul(row_bytes)
}

const BLOCK_FILE_MAGIC: [u8; 8] = *b"PXKVBLK1";
const BLOCK_FILE_VERSION: u32 = 1;
const PLANES_PER_LAYER: usize = 3;

/// Writes the block file for `header` and its planes into `out`, replacing its contents.
///
/// `planes` is the flat list `[k_even_0, k_odd_0, v_0, k_even_1, ...]`, three per layer, an
/// empty slice for an absent plane. Nothing is written to `out` when a refusal is returned.
pub fn encode_block(
    header: &BlockFileHeader,
    planes: &[&[f32]],
    out: &mut Vec<u8>,
) -> Result<(), InteropError> {
    check_planes(header, planes)?;
    out.clear();
    out.reserve(encoded_len(header));
    out.extend_from_slice(&BLOCK_FILE_MAGIC);
    out.extend_from_slice(&BLOCK_FILE_VERSION.to_le_bytes());
    out.extend_from_slice(&header.descriptor_digest);
    out.extend_from_slice(&header.content_key.to_le_bytes());
    out.extend_from_slice(&header.base_position.to_le_bytes());
    out.extend_from_slice(&(header.layers.len() as u32).to_le_bytes());
    header.layers.iter().for_each(|layer| write_layer_record(layer, out));
    planes
        .iter()
        .flat_map(|plane| plane.iter())
        .for_each(|value| out.extend_from_slice(&value.to_le_bytes()));
    Ok(())
}

fn write_layer_record(layer: &BlockFileLayer, out: &mut Vec<u8>) {
    [
        layer.rows,
        layer.k_even_row_bytes,
        layer.k_odd_row_bytes,
        layer.v_row_bytes,
        layer.ring_window,
        layer.ring_capacity,
    ]
    .iter()
    .for_each(|word| out.extend_from_slice(&word.to_le_bytes()));
}

fn check_planes(header: &BlockFileHeader, planes: &[&[f32]]) -> Result<(), InteropError> {
    let expected_planes = header.layers.len().checked_mul(PLANES_PER_LAYER);
    if expected_planes != Some(planes.len()) {
        return Err(InteropError::BlockFileMalformed { reason: "plane count is not 3 per layer" });
    }
    let row_bytes_per_plane = |layer: &BlockFileLayer| {
        [layer.k_even_row_bytes, layer.k_odd_row_bytes, layer.v_row_bytes]
    };
    let agrees = header.layers.iter().zip(planes.chunks(PLANES_PER_LAYER)).all(|(layer, chunk)| {
        chunk.iter().zip(row_bytes_per_plane(layer)).all(|(plane, row_bytes)| {
            plane.len().checked_mul(4)
                == (layer.rows as usize).checked_mul(row_bytes as usize)
        })
    });
    if agrees {
        Ok(())
    } else {
        Err(InteropError::BlockFileMalformed { reason: "plane length disagrees with the header" })
    }
}

/// A decoded block file: the parsed header and the plane bytes borrowed from the input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockFileView<'bytes> {
    /// The parsed header.
    pub header: BlockFileHeader,
    /// The bytes after the layer table, borrowed from the input.
    pub payload: &'bytes [u8],
}

/// Parses `bytes` as a block file, refusing any file whose length disagrees with its header.
pub fn decode_block(bytes: &[u8]) -> Result<BlockFileView<'_>, InteropError> {
    let fixed = bytes
        .get(..HEADER_FIXED_BYTES)
        .ok_or(malformed("shorter than the fixed header"))?;
    if fixed[0..8] != BLOCK_FILE_MAGIC {
        return Err(malformed("bad magic"));
    }
    if read_u32(fixed, 8) != BLOCK_FILE_VERSION {
        return Err(malformed("unsupported version"));
    }
    let table_end = (read_u32(fixed, 44) as usize)
        .checked_mul(LAYER_RECORD_BYTES)
        .and_then(|table| table.checked_add(HEADER_FIXED_BYTES))
        .ok_or(malformed("truncated layer table"))?;
    let table = bytes
        .get(HEADER_FIXED_BYTES..table_end)
        .ok_or(malformed("truncated layer table"))?;
    let mut descriptor_digest = [0u8; 16];
    descriptor_digest.copy_from_slice(&fixed[12..28]);
    let header = BlockFileHeader {
        descriptor_digest,
        content_key: read_u64(fixed, 28),
        base_position: read_u64(fixed, 36),
        layers: table.as_chunks::<LAYER_RECORD_BYTES>().0.iter().map(|record| read_layer_record(record)).collect(),
    };
    if bytes.len() != encoded_len(&header) {
        return Err(malformed("payload length disagrees with the header"));
    }
    Ok(BlockFileView { header, payload: &bytes[table_end..] })
}

impl BlockFileView<'_> {
    /// Refuses a block file whose descriptor digest is not `expected`.
    pub fn require_digest(&self, expected: [u8; 16]) -> Result<(), InteropError> {
        match self.header.descriptor_digest == expected {
            true => Ok(()),
            false => Err(InteropError::BlockFileDigestMismatch { expected, found: self.header.descriptor_digest }),
        }
    }
}

fn malformed(reason: &'static str) -> InteropError {
    InteropError::BlockFileMalformed { reason }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from(read_u32(bytes, offset)) | (u64::from(read_u32(bytes, offset + 4)) << 32)
}

fn read_layer_record(record: &[u8]) -> BlockFileLayer {
    BlockFileLayer {
        rows: read_u32(record, 0),
        k_even_row_bytes: read_u32(record, 4),
        k_odd_row_bytes: read_u32(record, 8),
        v_row_bytes: read_u32(record, 12),
        ring_window: read_u32(record, 16),
        ring_capacity: read_u32(record, 20),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn ramp(count: usize, start: f32) -> Vec<f32> {
        (0..count).map(|index| start + index as f32 * 0.5).collect()
    }

    fn fixture_header() -> BlockFileHeader {
        BlockFileHeader {
            descriptor_digest: [0x11; 16],
            content_key: 0x0102_0304_0506_0708,
            base_position: 64,
            layers: vec![
                BlockFileLayer { rows: 4, k_even_row_bytes: 8, k_odd_row_bytes: 8, v_row_bytes: 4, ring_window: 0, ring_capacity: 0 },
                BlockFileLayer { rows: 3, k_even_row_bytes: 8, k_odd_row_bytes: 8, v_row_bytes: 4, ring_window: 2, ring_capacity: 3 },
                BlockFileLayer::default(),
            ],
        }
    }

    #[test]
    fn blockfile_encoded_len_counts_header_table_and_planes() {
        let single = BlockFileHeader {
            layers: vec![BlockFileLayer { rows: 1, k_even_row_bytes: 4, k_odd_row_bytes: 4, v_row_bytes: 4, ring_window: 0, ring_capacity: 0 }],
            ..fixture_header()
        };
        let empty = BlockFileHeader { layers: Vec::new(), ..fixture_header() };

        assert_eq!(encoded_len(&fixture_header()), 260);
        assert_eq!(encoded_len(&empty), 48);
        assert_eq!(encoded_len(&single), 84);
        assert_eq!(ramp(3, 1.0), vec![1.0, 1.5, 2.0]);
    }

    #[test]
    fn blockfile_encoded_len_saturates_on_an_oversized_header() {
        let oversized = BlockFileHeader {
            layers: vec![BlockFileLayer {
                rows: u32::MAX,
                k_even_row_bytes: u32::MAX,
                k_odd_row_bytes: u32::MAX,
                v_row_bytes: u32::MAX,
                ring_window: 0,
                ring_capacity: 0,
            }],
            ..fixture_header()
        };

        assert_eq!(encoded_len(&oversized), usize::MAX);
    }

    fn words_le(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    fn fixture_planes() -> Vec<Vec<f32>> {
        vec![
            ramp(8, 0.0),
            ramp(8, 100.0),
            ramp(4, 200.0),
            ramp(6, 300.0),
            ramp(6, 400.0),
            ramp(3, 500.0),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ]
    }

    fn encode_fixture() -> Vec<u8> {
        let planes = fixture_planes();
        let views: Vec<&[f32]> = planes.iter().map(Vec::as_slice).collect();
        let mut out = Vec::new();
        encode_block(&fixture_header(), &views, &mut out).expect("encode the fixture");
        out
    }

    #[test]
    fn blockfile_encode_header_bytes() {
        let out = encode_fixture();

        assert_eq!(out[0..8], *b"PXKVBLK1");
        assert_eq!(out[8..12], [1, 0, 0, 0]);
        assert_eq!(out[12..28], [0x11; 16]);
        assert_eq!(out[28..36], [8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(out[36..44], [64, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(out[44..48], [3, 0, 0, 0]);
        assert_eq!(out[48..72], words_le(&[4, 8, 8, 4, 0, 0])[..]);
        assert_eq!(out[72..96], words_le(&[3, 8, 8, 4, 2, 3])[..]);
        assert_eq!(out[96..120], [0u8; 24]);
    }

    #[test]
    fn blockfile_encode_payload_layout() {
        let planes = fixture_planes();
        let views: Vec<&[f32]> = planes.iter().map(Vec::as_slice).collect();
        let mut out = encode_fixture();

        assert_eq!(out.len(), 260);
        assert_eq!(encoded_len(&fixture_header()), 260);
        assert_eq!(out[120..124], 0.0f32.to_le_bytes());
        assert_eq!(out[152..156], 100.0f32.to_le_bytes());
        assert_eq!(out[200..204], 300.0f32.to_le_bytes());
        assert_eq!(out[256..260], 501.0f32.to_le_bytes());
        encode_block(&fixture_header(), &views, &mut out).expect("encode the fixture");
        assert_eq!(out.len(), 260);
    }

    #[test]
    fn blockfile_encode_refuses_malformed_planes() {
        let mut planes = fixture_planes();
        let mut out = vec![0xAB; 5];
        planes[1].pop();
        let short: Vec<&[f32]> = planes.iter().map(Vec::as_slice).collect();
        let fewer: Vec<&[f32]> = short[..8].to_vec();

        let short_result = encode_block(&fixture_header(), &short, &mut out);
        assert!(matches!(
            short_result,
            Err(InteropError::BlockFileMalformed { reason: "plane length disagrees with the header" })
        ));
        assert_eq!(out, vec![0xAB; 5]);
        let fewer_result = encode_block(&fixture_header(), &fewer, &mut out);
        assert!(matches!(
            fewer_result,
            Err(InteropError::BlockFileMalformed { reason: "plane count is not 3 per layer" })
        ));
        assert_eq!(out, vec![0xAB; 5]);
    }

    #[test]
    fn blockfile_decode_reads_the_header_back() {
        let bytes = encode_fixture();

        let view = decode_block(&bytes).expect("decode the encoded fixture");

        assert_eq!(view.header, fixture_header());
        assert_eq!(view.payload.len(), 140);
    }

    #[test]
    fn blockfile_decode_refuses_short() {
        let result = decode_block(&[0u8; 47]);

        assert!(matches!(
            result,
            Err(InteropError::BlockFileMalformed { reason: "shorter than the fixed header" })
        ));
    }

    #[test]
    fn blockfile_decode_refuses_bad_magic() {
        let mut bytes = encode_fixture();
        bytes[0] = b'X';

        let result = decode_block(&bytes);

        assert!(matches!(result, Err(InteropError::BlockFileMalformed { reason: "bad magic" })));
    }

    #[test]
    fn blockfile_decode_refuses_bad_version() {
        let mut bytes = encode_fixture();
        bytes[8..12].copy_from_slice(&2u32.to_le_bytes());

        let result = decode_block(&bytes);

        assert!(matches!(
            result,
            Err(InteropError::BlockFileMalformed { reason: "unsupported version" })
        ));
    }

    #[test]
    fn blockfile_decode_refuses_truncated_layer_table() {
        let bytes = encode_fixture();

        let result = decode_block(&bytes[..100]);

        assert!(matches!(
            result,
            Err(InteropError::BlockFileMalformed { reason: "truncated layer table" })
        ));
    }

    #[test]
    fn blockfile_decode_refuses_wrong_payload_length() {
        let mut bytes = encode_fixture();
        let shorter = decode_block(&bytes[..bytes.len() - 4]);
        assert!(matches!(
            shorter,
            Err(InteropError::BlockFileMalformed { reason: "payload length disagrees with the header" })
        ));
        bytes.push(0);
        let longer = decode_block(&bytes);

        assert!(matches!(
            longer,
            Err(InteropError::BlockFileMalformed { reason: "payload length disagrees with the header" })
        ));
    }

    #[test]
    fn blockfile_digest_mismatch_refused() {
        let bytes = encode_fixture();
        let view = decode_block(&bytes).expect("decode the fixture");

        let refused = view.require_digest([0x22; 16]);

        assert!(matches!(
            refused,
            Err(InteropError::BlockFileDigestMismatch { expected, found })
                if expected == [0x22; 16] && found == [0x11; 16]
        ));
        assert!(matches!(view.require_digest([0x11; 16]), Ok(())));
    }
}
