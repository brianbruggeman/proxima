//! The typed reasons a backend-neutral render declines a layout.

/// Why [`crate::render`] could not spell a layout; each variant names the property that failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Refusal {
    /// The element neither fills whole bytes nor divides a byte, and is too wide to straddle two.
    #[error("a {width}-bit element has no flat byte address")]
    NoFlatAddress { width: u8 },
    /// An integer layout's power-of-two scale is not a value of the wide scalar.
    #[error("the integer scale is not a value of the wide scalar")]
    ScaleUnrepresentable,
    /// The hardware table holds no scalar to compute in.
    #[error("the hardware table has no wide scalar")]
    NoWideScalar,
}
