/// Errors from configuring or driving the persistent-memory primitives. A `Copy`
/// enum with no heap payload — error reporting allocates nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PmemError {
    /// The borrowed region is smaller than the configured layout requires.
    /// `need` is `8 + 2 * slot_len`; `got` is the region length supplied.
    #[error("region too small: need {need} bytes, got {got}")]
    RegionTooSmall {
        /// bytes the layout needs (root word plus two slots)
        need: usize,
        /// bytes the caller's region actually has
        got: usize,
    },
    /// A value handed to the update FSM is not exactly one slot wide. The CoW
    /// design copies the whole value into a fixed-size slot, so the lengths
    /// must match.
    #[error("value is {got} bytes, slot is {expected}")]
    SlotLenMismatch {
        /// the configured slot length
        expected: usize,
        /// the value length the caller supplied
        got: usize,
    },
    /// A slot length of zero was configured; a slot must hold at least one byte.
    #[error("slot length must be non-zero")]
    ZeroSlotLen,
}
