//! A draft model is a second [`LoadedModel`]: the same loader over a second GGUF,
//! gated on the target's vocabulary so the two models agree on what a token id means.
//!
//! [`LoadedModel::load`] builds a `ModelDescriptor` from a checkpoint's own metadata,
//! so a draft needs no per-family type; the only addition is the compatibility gate in
//! [`crate::draft_model_compatibility`], which mirrors llama.cpp's
//! `common_speculative_are_compatible`.
//!
//! The draft proposes candidate rows by driving its own serving loop
//! ([`LoadedModel::run_decode_loop_from_ids`], one `ServingState` per call) over the
//! committed history, so the proposer is a function `(history, limit) -> rows` that the
//! target's loop receives as a caller-supplied closure. [`LoadedModel::with_draft_model`]
//! is the bracket that loads the draft, hands that closure to the target's loop, and
//! releases the draft's plans, caches and weights in the order Metal residency requires
//! whether the loop returned or failed.

use core::ops::ControlFlow;

use proxima_tokenizer::sample::greedy_top_k_probability;

use crate::draft_model_compatibility::ensure_compatible;
use crate::error::InteropError;
use crate::serving::{SpeculativeConfig, SpeculativeType};

use super::ring_checkpoint::RingCheckpoint;
use super::*;

const DRAFT_CANDIDATE_COUNT: i32 = 10;

pub(super) type ModelProposer<'proposer> =
    dyn FnMut(&[u32], usize, &mut Vec<u32>) -> Result<(), InteropError> + 'proposer;

pub(super) fn map_draft_model(
    config: &SpeculativeConfig<'_>,
) -> Result<Option<Mmap>, InteropError> {
    if !config
        .speculative_types
        .contains(SpeculativeType::DraftSimple)
    {
        return Ok(None);
    }
    let path = config.draft_model_path.ok_or_else(|| {
        InteropError::UnsupportedServingConfig(String::from(
            "draft-simple requires draft_model_path",
        ))
    })?;
    let file = File::open(path)?;
    // model files remain immutable for the lifetime of both loaded models.
    Ok(Some(unsafe { Mmap::map(&file)? }))
}

fn greedy_config<'config>(serving_config: &ServingConfig<'config>) -> ServingConfig<'config> {
    let mut greedy = *serving_config;
    greedy.speculative = SpeculativeConfig::none();
    greedy.temperature = 0.0;
    greedy.repeat_penalty = 1.0;
    greedy.frequency_penalty = 0.0;
    greedy.presence_penalty = 0.0;
    greedy
}

fn advance(
    draft: &LoadedModel<'_>,
    runtime: &mut BackendRuntime,
    retained: &mut Option<PrefixState>,
    logits: &mut Vec<f32>,
    greedy: &ServingConfig<'_>,
    tokens: &[u32],
) -> Result<(), InteropError> {
    logits.clear();
    let (_, _, _, advanced) = draft.run_decode_loop_from_ids(
        tokens.to_vec(),
        1,
        greedy,
        runtime,
        None,
        &mut LogitsSink::Last(logits),
        &mut NodeValuesSink::Discard,
        &mut |_| ControlFlow::Continue(()),
        retained.take(),
        true,
        None,
        None,
    )?;
    *retained = Some(advanced);
    Ok(())
}

fn draft_chain_proposer<'draft>(
    draft: &'draft LoadedModel<'_>,
    runtime: &'draft mut BackendRuntime,
    retained: &'draft mut Option<PrefixState>,
    widths: &'draft [LayerPadRowWidths],
    serving_config: &ServingConfig<'draft>,
    target_vocab_size: usize,
    #[cfg(all(feature = "metal", target_os = "macos"))] residency: &'draft mut omega::metal::ResidentContext,
) -> impl FnMut(&[u32], usize, &mut Vec<u32>) -> Result<(), InteropError> + 'draft {
    let speculative = serving_config.speculative;
    let greedy = greedy_config(serving_config);
    let mut logits = Vec::with_capacity(draft.vocab.len());
    let mut candidates = Vec::with_capacity(draft.vocab.len());
    move |committed, limit, output| {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let _active = residency.activate();
        output.clear();
        // upstream checks the configured cap after pushing its first candidate.
        let limit = limit.min(speculative.n_max.max(1) as usize);
        if limit == 0 {
            return Ok(());
        }
        let cached = retained.as_ref().map_or(0, PrefixState::len);
        if cached >= committed.len()
            || retained
                .as_ref()
                .is_some_and(|state| !committed.starts_with(&state.ids))
        {
            return Err(InteropError::DraftModelState {
                reason: "committed tokens must extend the retained draft prefix",
            });
        }
        advance(draft, runtime, retained, &mut logits, &greedy, &committed[cached..])?;
        let checkpoint = retained
            .as_ref()
            .and_then(RingCheckpoint::capture_for_speculation)
            .ok_or(InteropError::DraftModelState {
                reason: "draft prefix cannot be checkpointed",
            })?;
        let chained = (|| {
            for position in 0..limit {
                let Some((token, probability)) =
                    greedy_top_k_probability(&logits, DRAFT_CANDIDATE_COUNT, &mut candidates)
                else {
                    return Err(InteropError::DraftModelState {
                        reason: "draft produced no candidates",
                    });
                };
                if probability < speculative.p_min {
                    break;
                }
                if token as usize >= target_vocab_size {
                    return Err(InteropError::DraftModelState {
                        reason: "draft token is outside the target vocabulary",
                    });
                }
                output.push(token);
                if position + 1 < limit {
                    advance(draft, runtime, retained, &mut logits, &greedy, &[token])?;
                }
            }
            Ok(())
        })();
        if let Some(state) = retained.as_mut() {
            state.restore_checkpoint(&checkpoint, widths);
        }
        chained?;
        // upstream casts its signed minimum to size_t before comparing lengths.
        if speculative.n_min < 0 || output.len() < speculative.n_min as usize {
            output.clear();
        }
        Ok(())
    }
}

impl LoadedModel<'_> {
    /// Loads `draft_mapping` as this model's draft and refuses it unless its vocabulary
    /// is compatible with this model's. Compose with [`Self::load`] directly when no
    /// compatibility is required; reach for this when the draft's tokens will be verified
    /// by this model.
    ///
    /// # Errors
    ///
    /// [`InteropError::IncompatibleDraftModelVocab`] when the vocabularies disagree at one
    /// of llama.cpp's predicates, otherwise whatever parsing or [`Self::load`] reports.
    pub fn load_draft<'draft>(
        &self,
        draft_mapping: &'draft [u8],
    ) -> Result<LoadedModel<'draft>, InteropError> {
        let target_metadata = proxima_gguf::parse_complete(self.checkpoint_mapping)?;
        let draft_metadata = proxima_gguf::parse_complete(draft_mapping)?;
        let draft = LoadedModel::load(&draft_metadata, draft_mapping)?;
        ensure_compatible(&target_metadata, &self.vocab, &draft_metadata, &draft.vocab)?;
        Ok(draft)
    }

    /// Runs `serve` with the proposer of the draft model `serving_config` names, or with
    /// `None` when it names none. The draft, its runtime and its retained cache outlive
    /// `serve` and are released after it returns or fails: plans first, then the cache,
    /// then the aliases of the resident weights, then the weights.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::load_draft`] or `serve` reports.
    pub(super) fn with_draft_model<Output>(
        &self,
        serving_config: &ServingConfig<'_>,
        serve: impl FnOnce(Option<&mut ModelProposer<'_>>) -> Result<Output, InteropError>,
    ) -> Result<Output, InteropError> {
        let Some(mapping) = map_draft_model(&serving_config.speculative)? else {
            return serve(None);
        };
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let mut residency = omega::metal::ResidentContext::default();
        let (draft, widths) = {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let _active = residency.activate();
            let draft = self.load_draft(&mapping)?;
            let (_, widths) = draft.declared_layer_cache_names_and_widths()?;
            (draft, widths)
        };
        let mut runtime = BackendRuntime::new(serving_config);
        let mut retained = None;
        let outcome = {
            let mut propose = draft_chain_proposer(
                &draft,
                &mut runtime,
                &mut retained,
                &widths,
                serving_config,
                self.vocab.len(),
                #[cfg(all(feature = "metal", target_os = "macos"))]
                &mut residency,
            );
            serve(Some(&mut propose))
        };
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let mut active = residency.activate();
        drop(runtime);
        drop(retained);
        #[cfg(all(feature = "metal", target_os = "macos"))]
        active.clear();
        drop(draft);
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::super::chunked_prefill_tests::dense_checkpoint;
    use super::*;

    fn loaded(checkpoint: &[u8]) -> LoadedModel<'_> {
        let parsed = proxima_gguf::pipe::parse_complete(checkpoint)
            .unwrap_or_else(|error| panic!("the dense fixture parses: {error:?}"));
        LoadedModel::load(&parsed, checkpoint)
            .unwrap_or_else(|error| panic!("the dense fixture loads: {error:?}"))
    }

    #[test]
    fn a_checkpoint_loads_as_its_own_draft() {
        let checkpoint = dense_checkpoint();
        let target = loaded(&checkpoint);

        let draft = target
            .load_draft(&checkpoint)
            .unwrap_or_else(|error| panic!("an identical vocabulary is compatible: {error:?}"));

        assert_eq!(draft.vocab.len(), target.vocab.len());
    }

    #[test]
    fn a_draft_with_a_diverging_vocabulary_is_refused_with_the_criterion() {
        let checkpoint = dense_checkpoint();
        let target = loaded(&checkpoint);
        let mut diverged = checkpoint.clone();
        let token_text = b"<eos-marker>";
        let position = diverged
            .windows(token_text.len())
            .position(|window| window == token_text)
            .unwrap_or_else(|| panic!("the fixture stores the eos marker token"));
        diverged[position + 9] = b'q';

        let error = target
            .load_draft(&diverged)
            .err()
            .unwrap_or_else(|| panic!("the marker at id 256 differs"));

        assert!(matches!(
            error,
            InteropError::IncompatibleDraftModelVocab {
                criterion: "token text",
                ..
            }
        ));
    }
}
