//! The arithmetic of one cached-attention op as data: a [`FloatLayout`] per stage plus the
//! blocking and normalisation choices, so a backend lowers each stage through the shared
//! proxima-numeric codec instead of choosing a rounding op per incumbent.
//!
//! Composes [`FloatLayout`] only; a stage held in layout `L` is `encode_with(.., L)` from the widest
//! layout and `decode(.., L)` back. Incumbent arithmetics are config rows built from these fields,
//! never Rust values.

use proxima_numeric::FloatLayout;
use proxima_numeric::float_layout::widest_executable_float;

/// Where the softmax scale multiplies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleSite {
    /// On the dot product, after the score is formed.
    AfterScore,
    /// On the query before the dot product, the scale held in `held`.
    OnQuery {
        /// Layout the scale is stored in before it multiplies the query.
        held: FloatLayout,
    },
}

/// How many keys one softmax or accumulate step consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyBlock {
    /// The lowering picks the block; only valid when no stage is narrower than `running`.
    Lowering,
    /// The incumbent's fixed block sizes.
    Fixed {
        /// Keys per softmax step.
        softmax: u32,
        /// Keys per accumulate step; equals the matrix product's k on a GPU.
        accumulate: u32,
    },
}

/// How the accumulated value is divided by the running sum at the end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Normalize {
    /// Multiply by `1 / sum`.
    Reciprocal,
    /// Divide by `sum`.
    Divide,
}

/// Why [`AttentionStages::new`] refused a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AttentionStagesError {
    /// [`KeyBlock::Lowering`] reorders accumulation, which is only exact when no stage is narrower
    /// than the running max and sum.
    #[error(
        "key block `Lowering` needs every stage at least as wide as running, `{stage}` is narrower"
    )]
    LoweringWithNarrowStage {
        /// The first stage found narrower than `running`.
        stage: &'static str,
    },
}

/// A [`FloatLayout`] per attention stage and the choices that are not a layout. A plan key, so it
/// is `Eq` and `Hash`; the offset and flush are binary32 bit patterns for that reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionStages {
    /// Query operand as the dot product reads it.
    pub query: FloatLayout,
    /// Key operand as the dot product reads it.
    pub key: FloatLayout,
    /// Value operand as the accumulate reads it.
    pub value: FloatLayout,
    /// The dot product, and the scaled score.
    pub score: FloatLayout,
    /// The softmax weight that multiplies a value row.
    pub probability: FloatLayout,
    /// The factor that rescales the accumulator when the max moves.
    pub rescale: FloatLayout,
    /// The value accumulator.
    pub accumulator: FloatLayout,
    /// The running max and running sum.
    pub running: FloatLayout,
    /// Where the scale multiplies.
    pub scale: ScaleSite,
    /// Keys per softmax and accumulate step.
    pub block: KeyBlock,
    /// Binary32 bits of an offset added to the max before the exponential, when set.
    pub max_offset_bits: Option<u32>,
    /// Binary32 bits of the exponent argument below which the exponential flushes to zero, when set.
    pub exp_flush_below_bits: Option<u32>,
    /// How the final division is done.
    pub normalize: Normalize,
}

impl AttentionStages {
    /// Every stage in the widest layout, the scale on the score, the lowering's own blocking, no
    /// offset, no flush, a reciprocal: the online softmax with no narrowing.
    pub const ONLINE: Self = Self {
        query: widest_executable_float(),
        key: widest_executable_float(),
        value: widest_executable_float(),
        score: widest_executable_float(),
        probability: widest_executable_float(),
        rescale: widest_executable_float(),
        accumulator: widest_executable_float(),
        running: widest_executable_float(),
        scale: ScaleSite::AfterScore,
        block: KeyBlock::Lowering,
        max_offset_bits: None,
        exp_flush_below_bits: None,
        normalize: Normalize::Reciprocal,
    };

    /// A checked value.
    ///
    /// # Errors
    /// [`AttentionStagesError::LoweringWithNarrowStage`] when `block` is [`KeyBlock::Lowering`] and a
    /// stage is narrower than `running`.
    #[expect(
        clippy::too_many_arguments,
        reason = "one argument per field, so a config row maps one to one"
    )]
    pub fn new(
        query: FloatLayout,
        key: FloatLayout,
        value: FloatLayout,
        score: FloatLayout,
        probability: FloatLayout,
        rescale: FloatLayout,
        accumulator: FloatLayout,
        running: FloatLayout,
        scale: ScaleSite,
        block: KeyBlock,
        max_offset_bits: Option<u32>,
        exp_flush_below_bits: Option<u32>,
        normalize: Normalize,
    ) -> Result<Self, AttentionStagesError> {
        let stages = Self {
            query,
            key,
            value,
            score,
            probability,
            rescale,
            accumulator,
            running,
            scale,
            block,
            max_offset_bits,
            exp_flush_below_bits,
            normalize,
        };
        stages.check()?;
        Ok(stages)
    }

    /// Re-runs the constructor's check on a value built from fields.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn check(&self) -> Result<(), AttentionStagesError> {
        if self.block != KeyBlock::Lowering {
            return Ok(());
        }
        let narrow = [
            ("query", self.query),
            ("key", self.key),
            ("value", self.value),
            ("score", self.score),
            ("probability", self.probability),
            ("rescale", self.rescale),
            ("accumulator", self.accumulator),
        ]
        .into_iter()
        .find(|(_, layout)| layout.width() < self.running.width());
        match narrow {
            Some((stage, _)) => Err(AttentionStagesError::LoweringWithNarrowStage { stage }),
            None => Ok(()),
        }
    }
}

impl Default for AttentionStages {
    fn default() -> Self {
        Self::ONLINE
    }
}

#[cfg(test)]
mod tests {
    use proxima_numeric::float_layout::half_width_float;
    use proxima_numeric::float_layout_codec::layout_supported;

    use super::*;

    fn with_narrow_accumulator(block: KeyBlock) -> Result<AttentionStages, AttentionStagesError> {
        let wide = AttentionStages::ONLINE;
        AttentionStages::new(
            wide.query,
            wide.key,
            wide.value,
            wide.score,
            wide.probability,
            wide.rescale,
            half_width_float(),
            wide.running,
            ScaleSite::AfterScore,
            block,
            None,
            None,
            Normalize::Reciprocal,
        )
    }

    #[test]
    fn attention_stages_refuse_lowering_with_a_stage_narrower_than_running() {
        assert_eq!(
            with_narrow_accumulator(KeyBlock::Lowering),
            Err(AttentionStagesError::LoweringWithNarrowStage {
                stage: "accumulator"
            })
        );
    }

    #[test]
    fn attention_stages_admit_a_narrow_stage_under_a_fixed_block() {
        let stages = with_narrow_accumulator(KeyBlock::Fixed {
            softmax: 1,
            accumulate: 1,
        })
        .expect("a fixed block admits a narrow accumulator");
        assert_eq!(stages.accumulator, half_width_float());
    }

    #[test]
    fn attention_stages_default_is_the_widest_online_value_and_passes_its_own_check() {
        let stages = AttentionStages::default();
        assert_eq!(stages, AttentionStages::ONLINE);
        assert_eq!(stages.check(), Ok(()));
        assert!(stages.accumulator.same_format(&stages.running));
        assert!(half_width_float().width() * 2 == stages.running.width());
        assert!(layout_supported(&stages.running).is_ok());
    }
}
