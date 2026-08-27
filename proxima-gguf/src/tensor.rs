//! One entry from the tensor directory (`gguf.h:17-23`): name, shape,
//! element type, and its byte offset into the (separately mmap'd or
//! read) data section.

use alloc::string::String;

use arrayvec::ArrayVec;

use crate::types::GgmlType;

/// Build-time floor constants — see `crate::sized` for the source of
/// truth and why these can never be runtime config.
pub use crate::sized::{MAX_DIMS, MAX_NAME_LEN};

/// One tensor's directory entry.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: ArrayVec<u64, MAX_DIMS>,
    pub ggml_type: GgmlType,
    /// Byte offset from the start of the data section (already validated
    /// contiguous-and-aligned by the parser — see
    /// `GgmlType::block_layout` and `gguf.cpp:622-636`'s `GGML_PAD` walk).
    pub offset: u64,
}

impl TensorInfo {
    /// Element count: product of `dims`, or 1 for a scalar (`dims` empty
    /// is not valid GGUF, but this stays total for any non-empty shape).
    #[must_use]
    pub fn element_count(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Exact byte footprint of this tensor's data, following the same
    /// block-quantization arithmetic as `ggml_nbytes` (`ggml.c`, driven by
    /// `type_traits[type].{blck_size,type_size}`, `ggml.c:1174,1178`).
    /// `None` on overflow (pathological dimensions from a corrupt file).
    #[must_use]
    pub fn nbytes(&self) -> Option<u64> {
        let layout = self.ggml_type.block_layout();
        let elements = self.element_count();
        let blocks = elements.checked_div(layout.block_elements)?;
        blocks.checked_mul(layout.block_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(values: &[u64]) -> ArrayVec<u64, MAX_DIMS> {
        values.iter().copied().collect()
    }

    #[test]
    fn element_count_multiplies_all_dims() {
        let tensor = TensorInfo {
            name: "blk.0.attn_q.weight".into(),
            dims: dims(&[4096, 4096, 1, 1]),
            ggml_type: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(tensor.element_count(), 4096 * 4096);
    }

    #[test]
    fn nbytes_accounts_for_quant_block_packing() {
        let tensor = TensorInfo {
            name: "blk.0.attn_q.weight".into(),
            dims: dims(&[32, 2, 1, 1]),
            ggml_type: GgmlType::Q4_0,
            offset: 0,
        };
        // 64 elements / 32-element blocks = 2 blocks * 18 bytes/block.
        assert_eq!(tensor.nbytes(), Some(36));
    }
}
