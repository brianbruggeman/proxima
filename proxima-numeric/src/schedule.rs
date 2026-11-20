//! The scheduler family a denoising loop is configured with: a [`Schedule`] (the sigma sequence)
//! and a [`StepRule`] (how one step moves the latent), both selected by config keys that name
//! the concept, never a model.
//!
//! Today the family holds `schedule = "flow_match"` ([`FlowMatchSchedule`]) and
//! `step = "euler"`. A beta-schedule family (epsilon or v prediction, DDPM-style) is a new
//! [`Schedule`] variant carrying its own parameter struct plus its math, and a DDIM step is a new
//! [`StepRule`] variant; neither changes a signature here, because both enums are
//! `non_exhaustive` and every consumer goes through [`Schedule::fill_sigmas`] and [`step`].
//!
//! Element type: values cross this module in the crate's `f32` pivot and are narrowed into a
//! [`FloatLayout`] the caller names, with `narrow_to_layout`'s rounding and overflow policy.
//!
//! Teaching pointer: [`euler_step`] on the graph is one `Op::Elementwise` expression,
//! `Add(x, Multiply(delta, v))` (`proxima-tensor` `ScalarOp::Add` / `ScalarOp::Multiply`); it
//! is linear in `x`, `v` and `delta`, so a reverse-mode transform needs only the add and
//! multiply rules (d/dx = 1, d/dv = delta, d/d delta = sum(v * upstream)).

use thiserror::Error;

use crate::convert::{Overflow, Rounding, narrow_to_layout};
use crate::error::NumericError;
use crate::float_layout::FloatLayout;
use crate::float_layout_codec::layout_supported;
use crate::flow_match::FlowMatchSchedule;

/// A schedule or step refused its input.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ScheduleError {
    /// The output slice does not hold the schedule's sigma count.
    #[error("output holds {found} sigmas; the schedule needs {expected}")]
    Length { found: usize, expected: usize },
    /// The compute layout is outside the scalar codec's domain.
    #[error(transparent)]
    Numeric(#[from] NumericError),
}

/// The sigma sequence, selected by the `schedule` config key.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
// an internally tagged enum buffers content on deserialize, which needs alloc
#[cfg_attr(feature = "alloc", derive(serde::Deserialize))]
#[serde(tag = "schedule", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Schedule {
    /// `schedule = "flow_match"`: a shifted flow-matching sigma sequence.
    FlowMatch(FlowMatchSchedule),
}

impl Schedule {
    /// Number of sigmas [`fill_sigmas`](Self::fill_sigmas) writes.
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::FlowMatch(inner) => inner.len(),
        }
    }

    /// A schedule always holds at least the leading sigma.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Write the sigmas into `out`.
    ///
    /// # Errors
    /// [`ScheduleError`] for a wrong-sized `out` or an unsupported compute layout.
    pub fn fill_sigmas(&self, out: &mut [f32]) -> Result<(), ScheduleError> {
        match self {
            Self::FlowMatch(inner) => inner.fill_sigmas(out),
        }
    }

    /// The sigmas as a new vector.
    ///
    /// # Errors
    /// As [`fill_sigmas`](Self::fill_sigmas).
    #[cfg(feature = "alloc")]
    pub fn sigmas(&self) -> Result<alloc::vec::Vec<f32>, ScheduleError> {
        match self {
            Self::FlowMatch(inner) => inner.sigmas(),
        }
    }
}

/// How one denoising step moves the latent, selected by the `step` config key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StepRule {
    /// `step = "euler"`: `x += (sigma_next - sigma) * v`, no noise injection.
    #[default]
    Euler,
}

/// Apply `rule` for one step from `sigma` to `sigma_next`, computing in `layout`.
///
/// # Errors
/// [`ScheduleError::Numeric`] when `layout` is outside the scalar codec's domain.
pub fn step(
    rule: StepRule,
    latent: &mut [f32],
    velocity: &[f32],
    sigma: f32,
    sigma_next: f32,
    layout: FloatLayout,
) -> Result<(), ScheduleError> {
    match rule {
        StepRule::Euler => euler_step(latent, velocity, sigma, sigma_next, layout),
    }
}

/// One Euler step: `x[i] = layout(x[i] + (sigma_next - sigma) * velocity[i])`.
/// Extra elements of the longer slice are left untouched.
///
/// # Errors
/// [`ScheduleError::Numeric`] when `layout` is outside the scalar codec's domain; the latent is
/// unchanged in that case.
pub fn euler_step(
    latent: &mut [f32],
    velocity: &[f32],
    sigma: f32,
    sigma_next: f32,
    layout: FloatLayout,
) -> Result<(), ScheduleError> {
    layout_supported(&layout)?;
    let delta = sigma_next - sigma;
    for (value, rate) in latent.iter_mut().zip(velocity) {
        *value = narrow(*value + delta * rate, layout)?;
    }
    Ok(())
}

pub(crate) fn narrow(value: f32, layout: FloatLayout) -> Result<f32, NumericError> {
    if layout == FloatLayout::BF32 {
        return Ok(value);
    }
    narrow_to_layout(value, layout, Rounding::NearestTiesEven, Overflow::Format)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn the_schedule_and_step_keys_name_the_concept_in_config() {
        let schedule = Schedule::FlowMatch(FlowMatchSchedule::new(4, 4096));
        let text = serde_json::to_string(&schedule).expect("serializes");
        assert!(text.contains(r#""schedule":"flow_match""#), "{text}");
        assert!(text.contains("mu_coefficients"), "{text}");
        let back: Schedule = serde_json::from_str(&text).expect("round trips");
        assert_eq!(back, schedule);
        assert_eq!(serde_json::to_string(&StepRule::Euler).as_deref().ok(), Some(r#""euler""#));
    }

    #[test]
    fn a_half_precision_layout_rounds_the_step_and_a_64_bit_layout_is_refused() {
        let mut latent = [1.0_f32];
        step(StepRule::Euler, &mut latent, &[1.0], 1.0, 0.99951, FloatLayout::BF16).expect("bf16");
        assert_eq!(latent[0], 1.0);
        let refused = euler_step(&mut latent, &[1.0], 1.0, 0.5, FloatLayout::F64);
        assert!(matches!(refused, Err(ScheduleError::Numeric(_))));
    }
}
