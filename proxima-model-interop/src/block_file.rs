//! Header types of the kv block file and the length of an encoded file.
//!
//! A block file is a header (fixed fields plus one record per layer) followed by the
//! per-layer `k_even`, `k_odd` and `v` planes. These types describe that header; the
//! length function says how many bytes a file with such a header occupies.

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
}
