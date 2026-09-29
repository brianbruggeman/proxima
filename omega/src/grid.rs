//! Thread-count arithmetic shared by every emitter ([`crate::msl`],
//! [`crate::wgsl`], [`crate::cuda`]): a grid's thread count is a product of
//! extents, and a product that wraps `u64` is the same silent truncation as a
//! grid wider than the launch API's index, one level up.

use proxima_tensor::NodeId;

use crate::error::EmitError;

/// The product of `factors`, or [`EmitError::GridExceedsThreadIndex`] with
/// `threads` saturated at `u64::MAX` when it does not fit.
///
/// # Errors
/// [`EmitError::GridExceedsThreadIndex`] when the product overflows `u64`.
pub(crate) fn checked_product(
    node: NodeId,
    factors: impl IntoIterator<Item = u64>,
) -> Result<u64, EmitError> {
    factors
        .into_iter()
        .try_fold(1u64, u64::checked_mul)
        .ok_or(EmitError::GridExceedsThreadIndex {
            node,
            threads: u64::MAX,
            limit: u64::MAX,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_product_that_fits_is_returned_exactly() {
        let product = checked_product(NodeId(17), [1873, 8960, 256]);

        assert_eq!(product, Ok(1873 * 8960 * 256));
    }

    #[test]
    fn an_empty_product_is_one_thread() {
        assert_eq!(checked_product(NodeId(0), []), Ok(1));
    }

    #[test]
    fn a_product_that_wraps_u64_is_a_named_error_not_a_truncated_count() {
        let overflow = checked_product(NodeId(17), [u64::from(u32::MAX), u64::from(u32::MAX), 8]);

        assert_eq!(
            overflow,
            Err(EmitError::GridExceedsThreadIndex {
                node: NodeId(17),
                threads: u64::MAX,
                limit: u64::MAX,
            })
        );
    }
}
