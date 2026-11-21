//! Launch geometry and thread-count arithmetic shared by every emitter
//! ([`crate::msl`], [`crate::wgsl`], [`crate::cuda`]): a grid's thread count is
//! a product of extents, and a product that wraps `u64` is the same silent
//! truncation as a grid wider than the launch API's index, one level up.

#[cfg(feature = "cuda")]
use core::num::NonZeroU64;

use proxima_tensor::NodeId;

use crate::error::EmitError;

/// Threads per block for every card-asm kernel whose [`LaunchGrid::block_width`]
/// is absent: the driver's choice, named here so the emit-time grid check and
/// the launch agree on how many blocks a grid needs.
#[cfg(feature = "cuda")]
pub const DEFAULT_BLOCK_WIDTH: u64 = 256;

/// Most blocks a card-asm launch's x dimension holds (`gridDim.x <= 2^31 - 1`).
/// A kernel's own `gid` is 64-bit, so this, not a 32-bit thread index, is the
/// widest grid an emitter covers.
#[cfg(feature = "cuda")]
pub(crate) const MAX_GRID_BLOCKS: u64 = i32::MAX as u64;

/// One card-asm launch. [`LaunchGrid::Idle`] is the only representation of an op
/// with no live output; a [`LaunchGrid::Launch`] always has at least one thread,
/// so no path can hand a driver the zero-block dispatch CUDA and HIP reject.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchGrid {
    Idle,
    Launch {
        threads: NonZeroU64,
        block_width: Option<u64>,
    },
}

#[cfg(feature = "cuda")]
impl LaunchGrid {
    /// A dispatch covering `threads`, or [`EmitError::ZeroThreadLaunch`] when
    /// an op with live output derives none. The bound-op layer owns whether an
    /// op launches at all ([`proxima_tensor::BoundOp::launches`]); this is the emitter's check
    /// that its own grid arithmetic agrees with that decision.
    ///
    /// # Errors
    /// [`EmitError::ZeroThreadLaunch`] when `threads` is zero.
    pub fn launch(
        node: NodeId,
        threads: u64,
        block_width: Option<u64>,
        kind: &'static str,
    ) -> Result<Self, EmitError> {
        let Some(threads) = NonZeroU64::new(threads) else {
            #[cfg(feature = "std")]
            proxima_telemetry::error!(
                node = node.0,
                kind,
                "a live op derived a zero-thread grid, which the launch API rejects"
            );
            return Err(EmitError::ZeroThreadLaunch { node, kind });
        };
        Ok(Self::Launch {
            threads,
            block_width,
        })
    }

    /// The op has no live output, so there is nothing to dispatch.
    #[must_use]
    pub const fn idle() -> Self {
        Self::Idle
    }

    #[must_use]
    pub const fn is_idle(self) -> bool {
        matches!(self, Self::Idle)
    }

    /// The launch's thread count; zero only for [`LaunchGrid::Idle`], which every
    /// launch path refuses before reading it.
    #[must_use]
    pub fn threads(self) -> u64 {
        match self {
            Self::Idle => 0,
            Self::Launch { threads, .. } => threads.get(),
        }
    }

    #[must_use]
    pub fn block_width(self) -> Option<u64> {
        match self {
            Self::Idle => None,
            Self::Launch { block_width, .. } => block_width,
        }
    }

    /// The block width this launch runs at: the pinned width, else
    /// [`DEFAULT_BLOCK_WIDTH`].
    #[must_use]
    pub fn block_width_or_default(self) -> u64 {
        self.block_width().unwrap_or(DEFAULT_BLOCK_WIDTH).max(1)
    }

    /// Blocks needed to cover [`LaunchGrid::threads`] at
    /// [`LaunchGrid::block_width_or_default`]; zero only for [`LaunchGrid::Idle`].
    #[must_use]
    pub fn blocks(self) -> u64 {
        self.threads().div_ceil(self.block_width_or_default())
    }
}

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
