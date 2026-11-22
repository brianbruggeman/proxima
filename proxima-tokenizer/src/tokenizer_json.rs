//! Loads a whole HuggingFace byte-level BPE `tokenizer.json` into a
//! ready-to-encode [`Vocab`]: `model.vocab`/`model.merges` (via
//! [`crate::hf`]'s readers), the `added_tokens` list, and the file's own
//! `pre_tokenizer` config.
//!
//! Added tokens ride the existing [`Vocab::with_token_types`] trie, so
//! [`crate::pipe::encode`] matches them whole before any pre-split or BPE
//! merge runs -- special tokens never merge. `special: true` entries are
//! tagged [`TokenType::Control`], the rest [`TokenType::UserDefined`].
//!
//! An added token whose id lies past `model.vocab` (HF appends them) extends
//! the vocab; its `content` is raw text and is spelled in the GPT-2
//! byte-level display alphabet ([`crate::byte_level::byte_to_char`]) so
//! [`Vocab::new`] decodes it back to the same bytes. One whose id is inside
//! `model.vocab` must already name the same token.
//!
//! The `pre_tokenizer` maps onto [`PreType`] only where the crate expresses
//! it exactly: `ByteLevel` with `use_regex` ([`PreType::Gpt2`]), or a
//! `Sequence` of individual-`Digits` then that `ByteLevel`
//! ([`PreType::DigitIsolatedGpt2`]). Anything else is a typed error, never a
//! silent fallback.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde_json::Value;

use crate::byte_level::byte_to_char;
use crate::error::TokenizerError;
use crate::hf::{merge_rules, tokens_by_id};
use crate::pretokenize::PreType;
use crate::vocab::{TokenType, Vocab};

/// Builds a [`Vocab`] from a `tokenizer.json`'s bytes (the caller reads the
/// file; this crate stays sans-IO). BOS/EOS/unknown ids are passed in for
/// the same reason [`crate::hf::vocab_from_tokenizer_json`] documents.
///
/// # Errors
///
/// [`TokenizerError::MalformedHfTokenizerJson`] for invalid JSON, a missing
/// or non-contiguous vocab, or an `added_tokens` entry that conflicts with
/// `model.vocab` or leaves a gap; [`TokenizerError::UnsupportedHfPreTokenizer`]
/// for a `pre_tokenizer` outside the set in the module doc; anything
/// [`Vocab::new`] can fail with.
pub fn load_tokenizer_json(
    bytes: &[u8],
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unknown_token_id: Option<u32>,
) -> Result<Vocab, TokenizerError> {
    let root: Value =
        serde_json::from_slice(bytes).map_err(|error| malformed(error.to_string()))?;
    let model = root
        .get("model")
        .ok_or_else(|| malformed("no top-level 'model' key"))?;
    let vocab_object = model
        .get("vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| malformed("model.vocab is missing or not a json object"))?;
    let merges_array = model
        .get("merges")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("model.merges is missing or not a json array"))?;

    let mut tokens = tokens_by_id(vocab_object)?;
    let mut types = alloc::vec![TokenType::Normal; tokens.len()];
    let added = added_tokens(root.get("added_tokens"))?;
    apply_added_tokens(&mut tokens, &mut types, added)?;

    let pre_type = pre_type_of(root.get("pre_tokenizer"))?;
    let merges = merge_rules(merges_array)?;
    Vocab::new(
        tokens,
        &merges,
        bos_token_id,
        eos_token_id,
        unknown_token_id,
    )?
    .with_token_types(types)
    .map(|vocab| vocab.with_pre_type(pre_type))
}

fn malformed(reason: impl Into<String>) -> TokenizerError {
    TokenizerError::MalformedHfTokenizerJson {
        reason: reason.into(),
    }
}

/// One `added_tokens` entry: its id, raw text, and `special` flag.
struct Added {
    id: usize,
    display: String,
    special: bool,
}

fn added_tokens(value: Option<&Value>) -> Result<BTreeMap<usize, Added>, TokenizerError> {
    let Some(array) = value.and_then(Value::as_array) else {
        return Ok(BTreeMap::new());
    };
    let mut by_id = BTreeMap::new();
    for entry in array {
        let id = entry
            .get("id")
            .and_then(Value::as_u64)
            .ok_or_else(|| malformed("added_tokens entry has no integer id"))?
            as usize;
        let content = entry
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed("added_tokens entry has no string content"))?;
        let special = entry
            .get("special")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let display = content.bytes().map(byte_to_char).collect();
        if by_id
            .insert(
                id,
                Added {
                    id,
                    display,
                    special,
                },
            )
            .is_some()
        {
            return Err(malformed(alloc::format!(
                "added_tokens id {id} appears twice"
            )));
        }
    }
    Ok(by_id)
}

fn apply_added_tokens(
    tokens: &mut Vec<String>,
    types: &mut Vec<TokenType>,
    added: BTreeMap<usize, Added>,
) -> Result<(), TokenizerError> {
    for (_, entry) in added {
        let tag = if entry.special {
            TokenType::Control
        } else {
            TokenType::UserDefined
        };
        match entry.id.cmp(&tokens.len()) {
            core::cmp::Ordering::Less if tokens[entry.id] == entry.display => types[entry.id] = tag,
            core::cmp::Ordering::Less => {
                return Err(malformed(alloc::format!(
                    "added_tokens id {} is {:?} but model.vocab names it {:?}",
                    entry.id,
                    entry.display,
                    tokens[entry.id]
                )));
            }
            core::cmp::Ordering::Equal => {
                tokens.push(entry.display);
                types.push(tag);
            }
            core::cmp::Ordering::Greater => {
                return Err(malformed(alloc::format!(
                    "added_tokens id {} leaves a gap after id {}",
                    entry.id,
                    tokens.len()
                )));
            }
        }
    }
    Ok(())
}

fn pre_type_of(value: Option<&Value>) -> Result<PreType, TokenizerError> {
    let unsupported = |kind: &str| TokenizerError::UnsupportedHfPreTokenizer {
        kind: kind.to_string(),
    };
    let Some(node) = value.filter(|node| !node.is_null()) else {
        return Err(unsupported("none (no pre_tokenizer)"));
    };
    match node.get("type").and_then(Value::as_str) {
        Some("ByteLevel") => byte_level_pre_type(node)
            .ok_or_else(|| unsupported("ByteLevel without use_regex or with add_prefix_space")),
        Some("Sequence") => sequence_pre_type(node)
            .ok_or_else(|| unsupported("Sequence other than Digits(individual) + ByteLevel")),
        Some(other) => Err(unsupported(other)),
        None => Err(unsupported("untyped pre_tokenizer")),
    }
}

fn byte_level_pre_type(node: &Value) -> Option<PreType> {
    let use_regex = node
        .get("use_regex")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let prefix_space = node
        .get("add_prefix_space")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    (use_regex && !prefix_space).then_some(PreType::Gpt2)
}

fn plain_byte_level(node: &Value) -> bool {
    node.get("type").and_then(Value::as_str) == Some("ByteLevel")
        && !node
            .get("add_prefix_space")
            .and_then(Value::as_bool)
            .unwrap_or(false)
}

fn sequence_pre_type(node: &Value) -> Option<PreType> {
    let steps = node.get("pretokenizers").and_then(Value::as_array)?;
    match steps.as_slice() {
        [digits, byte_level]
            if digits.get("type").and_then(Value::as_str) == Some("Digits")
                && digits.get("individual_digits").and_then(Value::as_bool) == Some(true)
                && byte_level.get("type").and_then(Value::as_str) == Some("ByteLevel") =>
        {
            byte_level_pre_type(byte_level).map(|_| PreType::DigitIsolatedGpt2)
        }
        [split, byte_level]
            if split.get("type").and_then(Value::as_str) == Some("Split")
                && split.get("behavior").and_then(Value::as_str) == Some("Isolated")
                && !split.get("invert").and_then(Value::as_bool).unwrap_or(false)
                && split
                    .get("pattern")
                    .and_then(|pattern| pattern.get("Regex"))
                    .and_then(Value::as_str)
                    == Some(crate::pretokenize::SINGLE_DIGIT_SPLIT_REGEX)
                && plain_byte_level(byte_level) =>
        {
            Some(PreType::SingleDigit)
        }
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipe::{decode, encode};

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../proxima-model-interop/tests/fixtures/mlx-reference/granite-3.1-1b-a400m-moe/plain/tokenizer.json"
    );

    fn load() -> Vocab {
        let bytes = std::fs::read(FIXTURE).expect("read the real tokenizer.json fixture");
        load_tokenizer_json(&bytes, None, Some(0), None).expect("real tokenizer.json loads")
    }

    #[test]
    fn byte_level_round_trip_on_real_text_uses_file_ids() {
        let vocab = load();
        let ids = encode("Hello world, the tokenizer", &vocab).expect("encode");
        assert_eq!(ids, [8279, 5788, 30, 322, 26764]);
        assert_eq!(
            decode(&ids, &vocab).expect("decode"),
            "Hello world, the tokenizer"
        );
    }

    #[test]
    fn special_tokens_stay_whole_and_never_merge() {
        let vocab = load();
        let ids = encode("Hello<|end_of_text|> world<fim_prefix>", &vocab).expect("encode");
        assert_eq!(ids, [8279, 0, 5788, 1]);
    }

    #[test]
    fn lowest_rank_merge_fires_first() {
        let vocab = load();
        let pairs = vocab.merge_pairs();
        let spelled = |rank: usize| {
            (
                vocab.token_str(pairs[rank].0),
                vocab.token_str(pairs[rank].1),
            )
        };
        assert_eq!(spelled(4), (Some("e"), Some("r")));
        assert_eq!(spelled(7), (Some("r"), Some("e")));
        assert_eq!(spelled(486), (Some("er"), Some("e")));
        let ids = encode("ere", &vocab).expect("encode");
        assert_eq!(
            ids,
            [747],
            "rank 4 (e,r) beats rank 7 (r,e); the other order would leave [87, 268]"
        );
    }

    #[test]
    fn added_token_beyond_model_vocab_has_its_file_id() {
        let vocab = load();
        assert_eq!(vocab.len(), 49155);
        assert_eq!(vocab.token_id("<|start_of_role|>"), Some(49152));
        let ids = encode("<|tool_call|>Hello", &vocab).expect("encode");
        assert_eq!(ids, [49154, 8279]);
        assert_eq!(vocab.token_str(49154), Some("<|tool_call|>"));
    }

    fn qwen_sequence(regex: &str) -> Value {
        serde_json::json!({
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": { "Regex": regex },
                    "behavior": "Isolated",
                    "invert": false
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": false,
                    "use_regex": false
                }
            ]
        })
    }

    #[test]
    fn qwen_split_then_byte_level_is_the_single_digit_pretokenizer() {
        let sequence = qwen_sequence(crate::pretokenize::SINGLE_DIGIT_SPLIT_REGEX);
        assert_eq!(sequence_pre_type(&sequence), Some(PreType::SingleDigit));
    }

    #[test]
    fn a_split_with_another_regex_is_refused_not_approximated() {
        let sequence = qwen_sequence(r"\p{L}+|\p{N}{1,3}");
        assert_eq!(sequence_pre_type(&sequence), None);
    }
}
