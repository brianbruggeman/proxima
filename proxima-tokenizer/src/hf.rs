//! Builds a [`Vocab`] from a real HuggingFace `tokenizer.json`'s own bytes.
//! Feature-gated (`hf`) for the same reason [`crate::gguf`] is: the
//! tokenizer core has no hard dependency on any one wire format's parser.
//!
//! Confirmed against the real, on-disk
//! `~/.lmstudio/models/HuggingFaceTB/SmolLM2-135M-Instruct/tokenizer.json`
//! (a `model.type == "BPE"` file, 49152-entry `model.vocab`, 48900-entry
//! `model.merges`): every added/special token (`<|im_start|>`, ...) is
//! already present in `model.vocab` at the same id `added_tokens` names, and
//! every id in `model.vocab` is contiguous over `0..vocab_size` -- exactly
//! what [`vocab_from_tokenizer_json`] assumes when it turns the JSON object
//! into an id-indexed `Vec<String>`.
//!
//! `model.merges` is already `"left right"` space-separated in the
//! GPT-2 byte-level display alphabet ([`crate::byte_level`]) -- the
//! identical shape [`Vocab::new`] already takes from GGUF's
//! `tokenizer.ggml.merges` array, so this module does no alphabet
//! conversion of its own, only JSON structure extraction.
//!
//! `bos_token_id`/`eos_token_id`/`unknown_token_id` are NOT read from
//! `tokenizer.json` itself -- HF spreads that fact across
//! `tokenizer_config.json`/`generation_config.json` instead, files this
//! sans-IO crate does not open. A caller passes them in explicitly, the same
//! contract [`Vocab::new`] itself already has.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde_json::Value;

use crate::error::TokenizerError;
use crate::vocab::Vocab;

/// Parses `bytes` (a `tokenizer.json` file's own bytes, however the caller
/// read them -- this crate stays sans-IO and never opens a file itself) and
/// builds a byte-level BPE [`Vocab`] from its `model.vocab`/`model.merges`.
///
/// # Errors
///
/// [`TokenizerError::MalformedHfTokenizerJson`] if `bytes` is not valid
/// JSON, or is missing `model.vocab`/`model.merges`, or `model.vocab`'s ids
/// are not exactly `0..vocab.len()` (this module's own contiguity
/// assumption, stated in the module doc). Anything [`Vocab::new`] can fail
/// with otherwise (a malformed merge rule, a missing base byte token).
pub fn vocab_from_tokenizer_json(
    bytes: &[u8],
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unknown_token_id: Option<u32>,
) -> Result<Vocab, TokenizerError> {
    let root: Value = serde_json::from_slice(bytes).map_err(|error| TokenizerError::MalformedHfTokenizerJson {
        reason: error.to_string(),
    })?;

    let model = root.get("model").ok_or_else(|| TokenizerError::MalformedHfTokenizerJson {
        reason: String::from("no top-level 'model' key"),
    })?;
    let vocab_object = model
        .get("vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| TokenizerError::MalformedHfTokenizerJson {
            reason: String::from("model.vocab is missing or not a json object"),
        })?;
    let merges_array = model
        .get("merges")
        .and_then(Value::as_array)
        .ok_or_else(|| TokenizerError::MalformedHfTokenizerJson {
            reason: String::from("model.merges is missing or not a json array"),
        })?;

    let tokens = tokens_by_id(vocab_object)?;
    let merges = string_array(merges_array, "model.merges")?;

    Vocab::new(tokens, &merges, bos_token_id, eos_token_id, unknown_token_id)
}

/// `model.vocab`'s `{token: id}` json object, re-indexed to an id-ordered
/// `Vec<String>` (index == token id) -- the shape [`Vocab::new`] takes,
/// mirroring [`crate::gguf::vocab_from_metadata`]'s own already-ordered
/// `tokenizer.ggml.tokens` array. Requires every id in `0..vocab_object.len()`
/// to be present exactly once; a real HF BPE tokenizer's own vocab is always
/// this dense (verified against the real SmolLM2 fixture in this module's
/// own doc), so a gap or duplicate here means the file is malformed rather
/// than something this reader should silently paper over.
fn tokens_by_id(vocab_object: &serde_json::Map<String, Value>) -> Result<Vec<String>, TokenizerError> {
    let mut tokens: Vec<Option<String>> = alloc::vec![None; vocab_object.len()];
    for (token, id_value) in vocab_object {
        let id = id_value.as_u64().ok_or_else(|| TokenizerError::MalformedHfTokenizerJson {
            reason: alloc::format!("model.vocab[{token:?}] is not an integer id"),
        })?;
        let slot = tokens
            .get_mut(id as usize)
            .ok_or_else(|| TokenizerError::MalformedHfTokenizerJson {
                reason: alloc::format!("model.vocab[{token:?}] = {id}, outside 0..{} (vocab is not id-contiguous)", vocab_object.len()),
            })?;
        if slot.replace(token.clone()).is_some() {
            return Err(TokenizerError::MalformedHfTokenizerJson {
                reason: alloc::format!("model.vocab id {id} is assigned to more than one token"),
            });
        }
    }
    tokens
        .into_iter()
        .enumerate()
        .map(|(id, token)| token.ok_or_else(|| TokenizerError::MalformedHfTokenizerJson {
            reason: alloc::format!("model.vocab has no token for id {id} (vocab is not id-contiguous)"),
        }))
        .collect()
}

fn string_array(values: &[Value], field: &'static str) -> Result<Vec<String>, TokenizerError> {
    values
        .iter()
        .map(|value| {
            value.as_str().map(String::from).ok_or_else(|| TokenizerError::MalformedHfTokenizerJson {
                reason: alloc::format!("{field} contains a non-string entry"),
            })
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A tiny, hand-built `tokenizer.json` shaped exactly like the real
    /// SmolLM2 fixture's own structure (a `model.vocab` object plus a
    /// `model.merges` array), covering the 256-byte alphabet plus one merge
    /// -- proves the JSON walk, not a synthetic bypass of it. Built through
    /// `serde_json::Value`, not hand-assembled string concatenation, so a
    /// display char that needs JSON escaping (control characters are part
    /// of this crate's own byte-level alphabet) still serializes correctly.
    fn tiny_tokenizer_json() -> alloc::string::String {
        let mut vocab_object = serde_json::Map::new();
        for byte in 0..=255u8 {
            let display = crate::byte_level::byte_to_char(byte).to_string();
            vocab_object.insert(display, Value::from(u64::from(byte)));
        }
        vocab_object.insert(String::from("hi"), Value::from(256u64));

        let root = serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": vocab_object,
                "merges": ["h i"],
            }
        });
        serde_json::to_string(&root).expect("constructed value serializes")
    }

    #[test]
    fn real_shaped_tokenizer_json_builds_a_vocab_with_the_merge_resolved() {
        let json = tiny_tokenizer_json();
        let vocab = vocab_from_tokenizer_json(json.as_bytes(), Some(1), Some(2), None)
            .expect("well-formed tokenizer.json builds a vocab");
        assert_eq!(vocab.len(), 257);
        assert_eq!(vocab.bos_token_id(), Some(1));
        assert_eq!(vocab.eos_token_id(), Some(2));
        assert_eq!(vocab.token_id("hi"), Some(256));
    }

    #[test]
    fn malformed_json_is_a_typed_error_not_a_panic() {
        let outcome = vocab_from_tokenizer_json(b"not json at all", None, None, None);
        assert!(matches!(outcome, Err(TokenizerError::MalformedHfTokenizerJson { .. })));
    }

    #[test]
    fn missing_model_vocab_is_a_typed_error_not_a_panic() {
        let outcome = vocab_from_tokenizer_json(br#"{"model": {"merges": []}}"#, None, None, None);
        assert!(matches!(outcome, Err(TokenizerError::MalformedHfTokenizerJson { .. })));
    }

    #[test]
    fn non_contiguous_vocab_ids_are_a_typed_error_not_a_panic() {
        let outcome = vocab_from_tokenizer_json(br#"{"model": {"vocab": {"a": 0, "b": 5}, "merges": []}}"#, None, None, None);
        assert!(matches!(outcome, Err(TokenizerError::MalformedHfTokenizerJson { .. })));
    }
}
