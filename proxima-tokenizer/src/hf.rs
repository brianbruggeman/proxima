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
//!
//! [`bos_eos_policy_from_tokenizer_config`] reads that other file
//! (`tokenizer_config.json`, not `tokenizer.json`) for the one fact it does
//! carry: whether the checkpoint's own config says to auto-add BOS/EOS.
//! Confirmed against the real, on-disk
//! `HuggingFaceTB/SmolLM2-135M-Instruct/tokenizer_config.json` (3,764
//! bytes) that the key can be entirely absent -- see that function's own
//! doc for why `Option<bool>`, not `bool`, is the honest shape here.

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

/// Reads `add_bos_token`/`add_eos_token` out of a real `tokenizer_config.json`
/// file's own bytes (a *different* file from `tokenizer.json` --
/// [`vocab_from_tokenizer_json`] never opens this one) and hands the result
/// straight to [`crate::vocab::Vocab::with_bos_eos_policy`].
///
/// `None` per field when the key is genuinely absent, distinct from
/// `Some(false)` -- confirmed load-bearing against the real, on-disk
/// `HuggingFaceTB/SmolLM2-135M-Instruct/tokenizer_config.json` (3,764
/// bytes): it carries `bos_token`/`eos_token` (the token *strings*) but has
/// **no `add_bos_token` key at all**. That absence is llama.cpp's own GGUF
/// conversion heuristic, not a fact this file states, so a caller must not
/// collapse "the file said nothing" into "the file said false".
///
/// # Errors
///
/// [`TokenizerError::MalformedHfTokenizerJson`] if `bytes` is not valid
/// JSON, or a present `add_bos_token`/`add_eos_token` key is not a JSON
/// boolean.
pub fn bos_eos_policy_from_tokenizer_config(bytes: &[u8]) -> Result<(Option<bool>, Option<bool>), TokenizerError> {
    let root: Value = serde_json::from_slice(bytes).map_err(|error| TokenizerError::MalformedHfTokenizerJson {
        reason: error.to_string(),
    })?;
    Ok((
        bool_field(&root, "add_bos_token")?,
        bool_field(&root, "add_eos_token")?,
    ))
}

/// `None` when `root.get(field)` is absent -- the whole point of
/// [`bos_eos_policy_from_tokenizer_config`], see its own doc.
fn bool_field(root: &Value, field: &'static str) -> Result<Option<bool>, TokenizerError> {
    match root.get(field) {
        None => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(TokenizerError::MalformedHfTokenizerJson {
            reason: alloc::format!("{field} is present but is not a json boolean"),
        }),
    }
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
    use std::path::Path;

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

    #[test]
    fn present_add_bos_and_add_eos_keys_read_as_some() {
        let (add_bos, add_eos) =
            bos_eos_policy_from_tokenizer_config(br#"{"add_bos_token": true, "add_eos_token": false}"#)
                .expect("well-formed tokenizer_config.json parses");
        assert_eq!(add_bos, Some(true));
        assert_eq!(add_eos, Some(false));
    }

    #[test]
    fn absent_add_bos_and_add_eos_keys_read_as_none_not_false() {
        let (add_bos, add_eos) = bos_eos_policy_from_tokenizer_config(br#"{"bos_token": "<|im_start|>"}"#)
            .expect("tokenizer_config.json missing the keys still parses");
        assert_eq!(add_bos, None, "an absent key must not be reported as Some(false)");
        assert_eq!(add_eos, None);
    }

    #[test]
    fn non_boolean_add_bos_token_is_a_typed_error_not_a_panic() {
        let outcome = bos_eos_policy_from_tokenizer_config(br#"{"add_bos_token": "yes"}"#);
        assert!(matches!(outcome, Err(TokenizerError::MalformedHfTokenizerJson { .. })));
    }

    #[test]
    fn malformed_tokenizer_config_json_is_a_typed_error_not_a_panic() {
        let outcome = bos_eos_policy_from_tokenizer_config(b"not json at all");
        assert!(matches!(outcome, Err(TokenizerError::MalformedHfTokenizerJson { .. })));
    }

    /// The real, on-disk `tokenizer_config.json` this crate's `add_bos_eos`
    /// module doc names -- confirms the genuine upstream quirk that started
    /// this work: SmolLM2's own config has no `add_bos_token` key at all
    /// (only `bos_token`, the token *string*), so `Option<bool>::None` is
    /// the only honest thing this reader can report. `#[ignore]`d: depends
    /// on a host-local model cache outside this repo.
    #[test]
    #[ignore = "depends on a host-local tokenizer_config.json checkout outside this repo"]
    fn real_smollm2_tokenizer_config_has_no_add_bos_token_key() {
        let path = Path::new(
            "/Users/brianbruggeman/.lmstudio/models/HuggingFaceTB/SmolLM2-135M-Instruct/tokenizer_config.json",
        );
        if !path.exists() {
            eprintln!("no real tokenizer_config.json found at {path:?}, skipping");
            return;
        }
        let bytes = std::fs::read(path).expect("read real smollm2 tokenizer_config.json");
        assert_eq!(bytes.len(), 3764, "the real file's own byte length -- confirms this isn't a stale copy");

        let (add_bos, add_eos) = bos_eos_policy_from_tokenizer_config(&bytes)
            .expect("real smollm2 tokenizer_config.json parses as json");
        assert_eq!(add_bos, None, "smollm2's real tokenizer_config.json has no add_bos_token key at all");
        assert_eq!(add_eos, None, "smollm2's real tokenizer_config.json has no add_eos_token key at all");

        let root: Value = serde_json::from_slice(&bytes).expect("real file is valid json");
        assert!(
            root.get("bos_token").is_some(),
            "sanity: the file does carry bos_token (the string), just not add_bos_token (the policy)"
        );
    }
}
