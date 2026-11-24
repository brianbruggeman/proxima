//! The one load-time prewarm every backend runs.
//!
//! [`LoadedModel::warm_at_load`] is the call a serving front end makes right
//! after `LoadedModel::load`, before it reports ready (llama-server's warm-up
//! decode is the incumbent behaviour). It composes the two generic steps with
//! no backend branch at the call site: [`LoadedModel::warm_resident_buffers`]
//! (the driver's own residency declaration, `0` where a driver has none) and
//! [`LoadedModel::warm_prefill`] (plan build and module load through the real
//! prefill path, so the first request finds both warm).

use super::*;
use crate::serving::ContextLength;

/// What [`LoadedModel::warm_at_load`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadWarm {
    /// Buffers the driver declared resident; `0` when it has nothing to declare.
    pub buffers: usize,
    /// Rows of the throwaway prompt `warm_prefill` ran.
    pub rows: usize,
}

/// Rows of one warm chunk: the configured micro-batch, capped so the
/// throwaway prompt (two chunks plus two decode rows) fits the served context.
pub(super) fn warm_chunk_rows(serving_config: &ServingConfig<'_>) -> usize {
    let ubatch = serving_config.ubatch_size as usize;
    let context_rows = match serving_config.context_length {
        ContextLength::Native => ubatch,
        ContextLength::Within(rows) | ContextLength::Extrapolate(rows) => rows as usize,
    };
    ubatch.min((context_rows.saturating_sub(2) / 2).max(1))
}

impl LoadedModel<'_> {
    /// Runs the residency warm then the prefill warm; see the module docs.
    ///
    /// # Errors
    ///
    /// Same as [`Self::warm_resident_buffers`] and [`Self::warm_prefill`].
    pub fn warm_at_load(&self, serving_config: &ServingConfig<'_>) -> Result<LoadWarm, InteropError> {
        let buffers = self.warm_resident_buffers(serving_config)?;
        let rows = self.warm_prefill(serving_config, warm_chunk_rows(serving_config))?;
        Ok(LoadWarm { buffers, rows })
    }
}
