//! Load-time warm-up of the prefill pipelines and resident plans.
//!
//! [`LoadedModel::warm_prefill`] is the call a serving front end makes after
//! [`LoadedModel::warm_resident_buffers`], before it takes a request. It runs a
//! throwaway prompt through the real chunked prefill and one greedy decode step
//! ([`LoadedModel::generate_from_ids`]), so the first request does not pay the
//! compile of the 512-row chunk pipelines or of the logits and greedy-head
//! pipelines, which `omega::metal::MetalDevice` keeps per device for the whole
//! process, nor the plan build for the chunk shapes, which
//! [`super::resident_plans`] keeps on the model's serving context.
//!
//! Contrast [`LoadedModel::prewarm`], which prefills real ids into the prompt
//! cache: this runs with the cache off, so no entry is stored and no KV state
//! outlives the call; the only things left behind are pipelines, resident
//! plans and the buffers a finished generation hands back, none of which a
//! later request's arithmetic reads.

use core::ops::ControlFlow;

use super::*;

/// Rows of the throwaway prompt that run after the first full chunk: a second
/// full chunk, so the last chunk is a chunk-width plan of the last-chunk shape
/// (it reads logits) that a shorter real tail refits instead of building.
const fn tail_rows(chunk_rows: usize) -> usize {
    chunk_rows
}

/// Token ids for a throwaway prompt of `rows` rows: a fixed pattern that
/// avoids id 0 and the end-of-sequence id, so the warm-up decode step runs.
fn throwaway_ids(rows: usize, vocab: &proxima_tokenizer::Vocab) -> Vec<u32> {
    let span = vocab.len().saturating_sub(1).max(1);
    let eos = vocab.eos_token_id();
    (0..rows)
        .map(|row| {
            let id = (row * 7 + 3) % span + 1;
            let id = u32::try_from(id).unwrap_or(1);
            if Some(id) == eos { id.saturating_add(1).min(u32::try_from(span).unwrap_or(1)) } else { id }
        })
        .collect()
}

impl LoadedModel<'_> {
    /// Compiles the prefill pipelines and builds the chunk plans now instead of
    /// in the first request. Runs `2 * chunk_rows` throwaway rows
    /// at `ubatch_size = chunk_rows` (one full non-final chunk, then a full last chunk),
    /// then one greedy decode step, with the prompt cache off. Returns the
    /// number of prompt rows it prefilled.
    ///
    /// Pass the `serving_config` requests will run under and the `chunk_rows`
    /// they chunk at (the `ubatch_size`): pipelines and plans are keyed by what
    /// the config selects, so a different config warms different ones.
    ///
    /// Each kept wide plan also fits its row template here (`Plan::prime_row_template`), so the
    /// short tail's refit binds nothing.
    ///
    /// Afterwards only what a request reuses stays resident: the two
    /// chunk-width plans (the non-final chunk, and the last chunk with logits) and
    /// the decode plan; the device KV buffers sized for the throwaway context are
    /// freed. A request whose last chunk is shorter refits the last-chunk plan to
    /// its row count (`Plan::refit_symbols`), so it builds no plan.
    ///
    /// The output of any later request is unchanged; the prompt cache holds no
    /// entry and the KV the throwaway prompt built is dropped with the call.
    ///
    /// Composes [`Self::generate_from_ids`] (the real prefill and decode path)
    /// with [`crate::PromptCacheConfig::off`]; compare
    /// [`Self::warm_resident_buffers`] (buffers only) and [`Self::prewarm`]
    /// (fills the prompt cache).
    ///
    /// # Errors
    ///
    /// Same as [`Self::generate_from_ids`].
    pub fn warm_prefill(
        &self,
        serving_config: &ServingConfig<'_>,
        chunk_rows: usize,
    ) -> Result<usize, InteropError> {
        if chunk_rows == 0 {
            return Ok(0);
        }
        let mut warm_config = *serving_config;
        warm_config.prompt_cache = crate::serving::PromptCacheConfig::off();
        warm_config.ubatch_size = u32::try_from(chunk_rows).unwrap_or(u32::MAX);
        let rows = chunk_rows + tail_rows(chunk_rows);
        let ids = throwaway_ids(rows, &self.vocab);
        self.generate_from_ids(&ids, 2, &warm_config, &mut |_| ControlFlow::Continue(()))?;
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        proxima_primitives::driver::block_on(self.serving.lock()).trim_warm_residue(chunk_rows)?;
        Ok(rows)
    }
}
