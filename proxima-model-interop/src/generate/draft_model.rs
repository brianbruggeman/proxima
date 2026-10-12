//! A draft model is a second [`LoadedModel`]: the same loader over a second GGUF,
//! gated on the target's vocabulary so the two models agree on what a token id means.
//!
//! [`LoadedModel::load`] builds a `ModelDescriptor` from a checkpoint's own metadata,
//! so a draft needs no per-family type; the only addition is the compatibility gate in
//! [`crate::draft_model_compatibility`], which mirrors llama.cpp's
//! `common_speculative_are_compatible`.

use crate::draft_model_compatibility::ensure_compatible;
use crate::error::InteropError;

use super::LoadedModel;

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
