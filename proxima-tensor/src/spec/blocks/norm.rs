use serde::{Deserialize, Serialize};

/// The statistic a [`Norm`] divides by.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormKind {
    /// Root mean square of the normalised axis: `x / sqrt(mean(x^2) + eps)`.
    #[default]
    Rms,
    /// Mean subtracted, then divided by the standard deviation.
    Layer,
}

/// Normalisation of a stream of rows, over the whole width or over contiguous
/// groups of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "config", derive(bon::Builder, conflaguration::Settings))]
#[cfg_attr(feature = "config", settings(prefix = "PROXIMA_NORM"))]
#[serde(deny_unknown_fields)]
pub struct Norm {
    /// Which statistic is divided out.
    #[serde(rename = "norm")]
    #[cfg_attr(feature = "config", setting(skip))]
    pub kind: NormKind,
    /// Added under the root of the variance, so a zero row stays finite (1e-5 in the llama family).
    pub eps: f32,
    /// Whether a learned per-channel gain multiplies the normalised rows; false is the unscaled form.
    pub scaled: bool,
    /// Normalise independently inside each contiguous group of this many channels (the head dimension
    /// for per-head norms); None normalises the whole width.
    #[cfg_attr(feature = "config", setting(skip))]
    pub group_width: Option<u32>,
}

#[cfg(feature = "config")]
impl conflaguration::Validate for Norm {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = alloc::vec::Vec::new();
        if !self.eps.is_finite() || self.eps <= 0.0 {
            errors.push(conflaguration::ValidationMessage::new(
                "eps",
                "must be positive and finite",
            ));
        }
        if self.group_width == Some(0) {
            errors.push(conflaguration::ValidationMessage::new(
                "group_width",
                "must be positive",
            ));
        }
        if errors.is_empty() {
            return Ok(());
        }
        Err(conflaguration::Error::Validation { errors })
    }
}
