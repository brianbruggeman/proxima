//! A flow-match sigma schedule and its Euler step.
//!
//! [`EmpiricalMu`] is the config value holding the four linear coefficients, the sequence-length
//! cutoff and the two anchor step counts of the empirical time-shift `mu`; [`FlowMatchSchedule`]
//! carries `steps`, `image_seq_len` and that value, and fills `steps + 1` sigmas
//! (`sigma[steps] = 0`). The denoiser is fed `t = sigma` as is: any timestep scaling or embedding
//! belongs to the network, not here.
//!
//! Element type: values cross this module in the crate's `f32` pivot (the same interchange
//! [`crate::Numeric`] and [`crate::convert::narrow_to_layout`] use), and the schedule carries the
//! compute [`FloatLayout`] as a config value, binary32 by default.
//! Every sigma and every updated element is narrowed into that layout, so bf16, f16 or a float8
//! compute is a config change, not a type change.
//! A layout the scalar codec cannot run (wider than 32 bits) is refused by name.
//!
//! The step rule lives in [`crate::schedule`], which also documents how the Euler update lowers to
//! one `Op::Elementwise` expression.
//!
//! Teaching pointer: [`FlowMatchSchedule::fill_sigmas`] writes into a caller-owned slice and
//! allocates nothing; [`FlowMatchSchedule::sigmas`] is the `alloc` convenience over it. The
//! exponential goes through `libm`, so the module builds at the no_std tier.

use crate::float_layout::FloatLayout;
use crate::schedule::{ScheduleError, narrow};

/// The empirical `mu` of the time shift, as config: `mu` is linear in the step count between two
/// anchors (`short_steps`, `long_steps`) whose values are linear in the sequence length, and past
/// `long_seq_len` it is the long-anchor line alone.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EmpiricalMu {
    /// Slope of the short-anchor value in the sequence length.
    pub short_slope: f32,
    /// Intercept of the short-anchor value.
    pub short_intercept: f32,
    /// Slope of the long-anchor value in the sequence length.
    pub long_slope: f32,
    /// Intercept of the long-anchor value.
    pub long_intercept: f32,
    /// Sequence lengths above this use the long-anchor value directly.
    pub long_seq_len: u32,
    /// Step count the short anchor is defined at.
    pub short_steps: f32,
    /// Step count the long anchor is defined at.
    pub long_steps: f32,
}

impl Default for EmpiricalMu {
    fn default() -> Self {
        Self {
            short_slope: 8.738_095_2e-5,
            short_intercept: 1.898_333_3,
            long_slope: 1.6927e-4,
            long_intercept: 0.456_666_66,
            long_seq_len: 4300,
            short_steps: 10.0,
            long_steps: 200.0,
        }
    }
}

impl EmpiricalMu {
    /// `mu` for a sequence length and a step count.
    #[must_use]
    pub fn mu(&self, image_seq_len: u32, steps: u32) -> f32 {
        let length = image_seq_len as f32;
        let long = self.long_slope * length + self.long_intercept;
        if image_seq_len > self.long_seq_len {
            return long;
        }
        let short = self.short_slope * length + self.short_intercept;
        let slope = (long - short) / (self.long_steps - self.short_steps);
        let intercept = long - self.long_steps * slope;
        slope * steps as f32 + intercept
    }
}

/// `steps` denoising steps over `image_seq_len` tokens, shifted by [`EmpiricalMu`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FlowMatchSchedule {
    pub steps: u32,
    pub image_seq_len: u32,
    #[serde(default)]
    pub mu_coefficients: EmpiricalMu,
    /// Layout the sigmas are computed in; binary32 unless configured otherwise.
    #[serde(default = "binary32")]
    pub layout: FloatLayout,
}

impl FlowMatchSchedule {
    /// A schedule with the default [`EmpiricalMu`].
    #[must_use]
    pub fn new(steps: u32, image_seq_len: u32) -> Self {
        let mu_coefficients = EmpiricalMu::default();
        Self { steps, image_seq_len, mu_coefficients, layout: FloatLayout::BF32 }
    }

    /// The `mu` this schedule shifts by.
    #[must_use]
    pub fn mu(&self) -> f32 {
        self.mu_coefficients.mu(self.image_seq_len, self.steps)
    }

    /// Number of sigmas [`fill_sigmas`](Self::fill_sigmas) writes: `steps + 1`, or 1 for no steps.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.steps as usize + 1
    }

    /// A schedule always holds at least the leading sigma.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Write the sigmas into `out`, which must hold [`len`](Self::len) values.
    ///
    /// # Errors
    /// [`ScheduleError::Length`] for a wrong-sized `out`; [`ScheduleError::Numeric`] when the
    /// compute layout is outside the scalar codec's domain.
    pub fn fill_sigmas(&self, out: &mut [f32]) -> Result<(), ScheduleError> {
        if out.len() != self.len() {
            return Err(ScheduleError::Length { found: out.len(), expected: self.len() });
        }
        let shift = libm::expf(self.mu());
        let steps = self.steps.max(1) as f32;
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = time_shift(shift, 1.0 - index as f32 / steps);
        }
        out[0] = 1.0;
        if self.steps > 0 {
            out[self.steps as usize] = 0.0;
        }
        for slot in out.iter_mut() {
            *slot = narrow(*slot, self.layout)?;
        }
        Ok(())
    }

    /// The sigmas as a new vector.
    ///
    /// # Errors
    /// As [`fill_sigmas`](Self::fill_sigmas).
    #[cfg(feature = "alloc")]
    pub fn sigmas(&self) -> Result<alloc::vec::Vec<f32>, ScheduleError> {
        let mut out = alloc::vec![1.0_f32; self.len()];
        self.fill_sigmas(&mut out)?;
        Ok(out)
    }
}

fn time_shift(shift: f32, fraction: f32) -> f32 {
    if fraction >= 1.0 {
        1.0
    } else if fraction <= 0.0 {
        0.0
    } else {
        shift / (shift + (1.0 / fraction - 1.0))
    }
}

const fn binary32() -> FloatLayout {
    FloatLayout::BF32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_long_sequence_branch_starts_strictly_above_the_cutoff() {
        let config = EmpiricalMu::default();
        let long = |length: f32| config.long_slope * length + config.long_intercept;
        assert_eq!(config.mu(4301, 4), long(4301.0));
        assert_ne!(config.mu(4300, 4), long(4300.0));
    }

    #[test]
    fn at_the_long_anchor_step_count_both_branches_agree() {
        let config = EmpiricalMu::default();
        let at_cutoff = config.mu(4300, 200);
        let long = config.long_slope * 4300.0 + config.long_intercept;
        assert!((at_cutoff - long).abs() < 1e-4, "{at_cutoff} vs {long}");
    }

    #[test]
    fn the_schedule_runs_from_one_to_zero_and_never_rises() {
        let sigmas = FlowMatchSchedule::new(8, 1024).sigmas().expect("binary32 is supported");
        assert_eq!(sigmas.len(), 9);
        assert_eq!((sigmas[0], sigmas[8]), (1.0, 0.0));
        assert!(sigmas.windows(2).all(|pair| pair[0] > pair[1]));
    }

    #[test]
    fn the_time_shift_edges_are_exact() {
        assert_eq!(time_shift(9.9, 1.0), 1.0);
        assert_eq!(time_shift(9.9, 1.5), 1.0);
        assert_eq!(time_shift(9.9, 0.0), 0.0);
        assert_eq!(time_shift(9.9, -0.5), 0.0);
    }

    #[test]
    fn zero_steps_is_the_single_leading_sigma_and_a_wrong_length_is_refused() {
        let schedule = FlowMatchSchedule::new(0, 4096);
        assert_eq!(schedule.sigmas(), Ok(std::vec![1.0]));
        let mut short = [0.0_f32; 2];
        let refusal = ScheduleError::Length { found: 2, expected: 1 };
        assert_eq!(schedule.fill_sigmas(&mut short), Err(refusal));
    }
}
