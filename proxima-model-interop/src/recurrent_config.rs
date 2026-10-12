//! The graph-bound recurrence policy a model is loaded with and a serving call
//! must agree with.
//!
//! [`RecurrentConfig`] is plain `Copy` data, like the rest of
//! [`crate::ServingConfig`]: it names how many times a checkpoint's physical
//! layers run and how the completed passes are contrasted, and
//! [`RecurrentSettings`](crate::RecurrentSettings) is its env/TOML/fluent
//! mirror. The policy changes the lowered graph, not just a runtime knob, so a
//! model resolves it once at load ([`RecurrentConfig::resolve`]) and serving
//! admission compares each call's config against that resolved identity before
//! any plan or cache work ([`RecurrentConfig::matches_model`]).
//!
//! The graph itself is `proxima_tensor::spec`: `RecurrenceConfig` and
//! `LoopCdConfig` on a `ModelDescriptor`, lowered by `build_forward`.

use proxima_tensor::LoopCdMode;

use crate::error::InteropError;

/// Recurrent graph identity selected at load and checked on every serving call.
/// `iterations: None` resolves from the checkpoint's own metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecurrentConfig {
    /// Completed passes over the physical schedule; `None` takes the
    /// checkpoint's depth (a GGUF `{arch}.num_loops`, an HF `total_ut_steps`).
    pub iterations: Option<u32>,
    /// Which completed-pass representation is contrasted.
    pub mode: LoopCdMode,
    /// Contrast strength, or its ceiling when `mode` is adaptive.
    pub strength: f32,
    /// One-based completed pass the terminal pass is contrasted against.
    pub reference_iteration: u32,
}

impl RecurrentConfig {
    /// Ordinary execution: the checkpoint's own schedule, guidance disabled.
    #[must_use]
    pub const fn neutral() -> Self {
        Self {
            iterations: None,
            mode: LoopCdMode::Disabled,
            strength: 0.0,
            reference_iteration: 1,
        }
    }

    /// Fills `iterations` from the checkpoint's depth and validates the whole
    /// policy against it.
    ///
    /// # Errors
    ///
    /// [`InteropError::InvalidRecurrentConfig`] for zero iterations, a strength
    /// that is not finite and nonnegative, a reference pass outside
    /// `1..=iterations`, or a disabled mode that still carries strength.
    pub fn resolve(self, checkpoint_iterations: u32) -> Result<Self, InteropError> {
        let iterations = self.iterations.unwrap_or(checkpoint_iterations);
        let invalid = |reason| Err(InteropError::InvalidRecurrentConfig { reason });
        if iterations == 0 {
            return invalid("iterations must be positive");
        }
        if !self.strength.is_finite() || self.strength < 0.0 {
            return invalid("strength must be finite and nonnegative");
        }
        if self.reference_iteration == 0 || self.reference_iteration > iterations {
            return invalid("reference_iteration must be within resolved iterations");
        }
        if self.mode == LoopCdMode::Disabled && self.strength != 0.0 {
            return invalid("disabled LoopCD requires zero strength");
        }
        Ok(Self {
            iterations: Some(iterations),
            ..self
        })
    }

    /// `true` when this policy asks for nothing beyond the ordinary schedule.
    #[must_use]
    pub const fn is_neutral(self) -> bool {
        self.iterations.is_none()
            && matches!(self.mode, LoopCdMode::Disabled)
            && self.strength == 0.0
            && self.reference_iteration == 1
    }

    /// Whether a serving call's `requested` policy names the graph this model
    /// resolved at load. A request that leaves `iterations` open means the
    /// model's own depth. Strength compares by bits so a NaN request never
    /// matches and `-0.0` is not `0.0`.
    #[must_use]
    pub fn matches_model(self, requested: Self) -> bool {
        let requested = Self {
            iterations: requested.iterations.or(self.iterations),
            ..requested
        };
        self.iterations == requested.iterations
            && self.mode == requested.mode
            && self.strength.to_bits() == requested.strength.to_bits()
            && self.reference_iteration == requested.reference_iteration
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn guided(iterations: Option<u32>, mode: LoopCdMode, strength: f32) -> RecurrentConfig {
        RecurrentConfig {
            iterations,
            mode,
            strength,
            reference_iteration: 1,
        }
    }

    #[test]
    fn open_depth_resolves_to_the_checkpoint_depth() {
        let open = guided(None, LoopCdMode::FixedLogits, 0.5);
        let explicit = guided(Some(4), LoopCdMode::FixedLogits, 0.5);

        assert_eq!(
            open.resolve(4).expect("open resolves"),
            explicit.resolve(4).expect("explicit resolves")
        );
        assert_eq!(open.resolve(2).expect("open resolves").iterations, Some(2));
        assert_eq!(
            explicit.resolve(2).expect("explicit wins").iterations,
            Some(4)
        );
    }

    #[test]
    fn unusable_policies_are_refused_by_name() {
        let refused = |config: RecurrentConfig, depth: u32| {
            matches!(
                config.resolve(depth),
                Err(InteropError::InvalidRecurrentConfig { .. })
            )
        };

        assert!(refused(guided(Some(0), LoopCdMode::Disabled, 0.0), 4));
        assert!(refused(guided(None, LoopCdMode::Disabled, 0.0), 0));
        assert!(refused(
            guided(Some(2), LoopCdMode::FixedLogits, f32::NAN),
            4
        ));
        assert!(refused(guided(Some(2), LoopCdMode::FixedLogits, -0.5), 4));
        assert!(refused(
            guided(Some(2), LoopCdMode::FixedLogits, f32::INFINITY),
            4
        ));
        assert!(refused(guided(Some(2), LoopCdMode::Disabled, 0.5), 4));
        assert!(refused(
            RecurrentConfig {
                reference_iteration: 3,
                ..guided(Some(2), LoopCdMode::FixedHidden, 0.5)
            },
            4
        ));
        assert!(refused(
            RecurrentConfig {
                reference_iteration: 0,
                ..guided(Some(2), LoopCdMode::FixedHidden, 0.5)
            },
            4
        ));
    }

    #[test]
    fn a_model_admits_only_the_policy_it_resolved() {
        let model = guided(None, LoopCdMode::FixedLogits, 0.5)
            .resolve(4)
            .expect("resolves");

        assert!(model.matches_model(guided(None, LoopCdMode::FixedLogits, 0.5)));
        assert!(model.matches_model(guided(Some(4), LoopCdMode::FixedLogits, 0.5)));
        assert!(!model.matches_model(guided(Some(2), LoopCdMode::FixedLogits, 0.5)));
        assert!(!model.matches_model(guided(Some(4), LoopCdMode::FixedHidden, 0.5)));
        assert!(!model.matches_model(guided(Some(4), LoopCdMode::FixedLogits, 0.25)));
        assert!(!model.matches_model(guided(Some(4), LoopCdMode::FixedLogits, f32::NAN)));
        assert!(!model.matches_model(RecurrentConfig {
            reference_iteration: 2,
            ..guided(Some(4), LoopCdMode::FixedLogits, 0.5)
        }));
    }

    #[test]
    fn neutral_is_the_ordinary_schedule() {
        assert!(RecurrentConfig::neutral().is_neutral());
        assert!(!guided(Some(4), LoopCdMode::Disabled, 0.0).is_neutral());
        assert!(!guided(None, LoopCdMode::FixedLogits, 0.5).is_neutral());
    }
}
