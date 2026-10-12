//! Writes a [`Vocab`] back out as the two files a HuggingFace tokenizer
//! directory carries -- the inverse of [`crate::hf`]. Feature-gated (`hf`)
//! with the reader, sans-IO: both functions return the file's bytes and
//! the caller decides where they land.
//!
//! Only what a [`Vocab`] holds is written: `model.vocab`, `model.merges`
//! (the `["left", "right"]` pair spelling, priority order), `added_tokens`
//! for every [`TokenType::Control`]/[`TokenType::UserDefined`] entry, and
//! in the config the bos/eos/unk token strings plus the auto-add policy.
//! The `normalizer`, `pre_tokenizer`, `post_processor` and `decoder` blocks
//! and the extra `model` keys follow the vocab's own shape, as
//! `crate::hf_write_pipeline` describes: a char-level vocab gets the
//! SentencePiece-style blocks, a byte-level vocab the ByteLevel blocks its
//! [`crate::pretokenize::PreType`] selects.

use alloc::string::String;
use alloc::vec::Vec;

use serde_json::{Map, Value, json};

use crate::error::TokenizerError;
use crate::hf_write_pipeline::pipeline_of;
use crate::vocab::{TokenType, Vocab};

/// Serializes `vocab` to a `tokenizer.json` whose `model.vocab` and
/// `model.merges` [`crate::hf::vocab_from_tokenizer_json`] reads back into
/// an equal [`Vocab`] (ids passed by the caller, token types restored from
/// `added_tokens`).
///
/// Compose this with [`tokenizer_config_json_bytes`] for the second file;
/// the vocab type tags and ids are the only inputs, so nothing here
/// touches a file.
///
/// # Errors
///
/// [`TokenizerError::HfTokenizerJsonWrite`] if two ids carry the same
/// token string (a JSON object cannot hold both) or serialization fails.
pub fn tokenizer_json_bytes(vocab: &Vocab) -> Result<Vec<u8>, TokenizerError> {
    let pipeline = pipeline_of(vocab)?;
    let mut model = json!({
        "type": "BPE",
        "vocab": vocab_object(vocab)?,
        "merges": merges_array(vocab),
    });
    if let (Some(entries), Some(extras)) =
        (model.as_object_mut(), pipeline.model_extras.as_object())
    {
        entries.extend(extras.clone());
    }
    let root = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens(vocab),
        "normalizer": pipeline.normalizer,
        "pre_tokenizer": pipeline.pre_tokenizer,
        "post_processor": pipeline.post_processor,
        "decoder": pipeline.decoder,
        "model": model,
    });
    to_bytes(&root)
}

/// Serializes the facts a `tokenizer_config.json` states about `vocab`:
/// `bos_token`, `eos_token`, `unk_token` (strings, omitted when the vocab
/// has no such id) and `add_bos_token`, `add_eos_token` (omitted when the
/// vocab's policy is `None`, which [`crate::hf::bos_eos_policy_from_tokenizer_config`]
/// reads back as `None`, never `false`).
///
/// # Errors
///
/// [`TokenizerError::HfTokenizerJsonWrite`] if serialization fails.
pub fn tokenizer_config_json_bytes(vocab: &Vocab) -> Result<Vec<u8>, TokenizerError> {
    let mut config = Map::new();
    insert_token(&mut config, "bos_token", vocab, vocab.bos_token_id());
    insert_token(&mut config, "eos_token", vocab, vocab.eos_token_id());
    insert_token(&mut config, "unk_token", vocab, vocab.unknown_token_id());
    insert_flag(&mut config, "add_bos_token", vocab.add_bos_token());
    insert_flag(&mut config, "add_eos_token", vocab.add_eos_token());
    to_bytes(&Value::Object(config))
}

fn to_bytes(root: &Value) -> Result<Vec<u8>, TokenizerError> {
    serde_json::to_vec(root).map_err(|error| TokenizerError::HfTokenizerJsonWrite {
        reason: alloc::string::ToString::to_string(&error),
    })
}

fn vocab_object(vocab: &Vocab) -> Result<Map<String, Value>, TokenizerError> {
    let object: Map<String, Value> = (0..vocab.len())
        .filter_map(|id| u32::try_from(id).ok())
        .filter_map(|id| {
            vocab
                .token_str(id)
                .map(|token| (String::from(token), json!(id)))
        })
        .collect();
    if object.len() == vocab.len() {
        return Ok(object);
    }
    Err(TokenizerError::HfTokenizerJsonWrite {
        reason: alloc::format!(
            "{} ids but {} distinct token strings",
            vocab.len(),
            object.len()
        ),
    })
}

fn merges_array(vocab: &Vocab) -> Vec<Value> {
    vocab
        .merge_pairs()
        .into_iter()
        .filter_map(|(left, right)| Some(json!([vocab.token_str(left)?, vocab.token_str(right)?])))
        .collect()
}

fn added_tokens(vocab: &Vocab) -> Vec<Value> {
    (0..vocab.len())
        .filter_map(|id| u32::try_from(id).ok())
        .filter_map(|id| added_token(vocab, id))
        .collect()
}

fn added_token(vocab: &Vocab, id: u32) -> Option<Value> {
    let special = match vocab.token_type(id)? {
        TokenType::Control => true,
        TokenType::UserDefined => false,
        _ => return None,
    };
    Some(json!({
        "id": id,
        "content": vocab.token_str(id)?,
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": !special,
        "special": special,
    }))
}

fn insert_token(config: &mut Map<String, Value>, key: &str, vocab: &Vocab, id: Option<u32>) {
    if let Some(token) = id.and_then(|id| vocab.token_str(id)) {
        config.insert(String::from(key), json!(token));
    }
}

fn insert_flag(config: &mut Map<String, Value>, key: &str, flag: Option<bool>) {
    if let Some(flag) = flag {
        config.insert(String::from(key), json!(flag));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::sync::OnceLock;

    use super::*;
    use crate::hf::{bos_eos_policy_from_tokenizer_config, vocab_from_tokenizer_json};
    use crate::pretokenize::PreType;

    const FIXTURES: &str = "../proxima-model-interop/tests/fixtures/mlx-reference";

    struct Reference {
        json: Value,
        config: Value,
        plain: Vocab,
        typed: Vocab,
        written_json: Value,
        written_config: Value,
    }

    fn special_ids(json: &Value, config: &Value) -> [Option<u32>; 3] {
        ["bos_token", "eos_token", "unk_token"].map(|key| {
            let token = config.get(key)?.as_str()?;
            u32::try_from(json["model"]["vocab"].get(token)?.as_u64()?).ok()
        })
    }

    fn token_types(json: &Value, len: usize) -> Vec<TokenType> {
        let mut types = alloc::vec![TokenType::Normal; len];
        for added in json["added_tokens"].as_array().expect("added_tokens array") {
            let id = usize::try_from(added["id"].as_u64().expect("added id")).expect("id fits");
            let Some(slot) = types.get_mut(id) else {
                continue;
            };
            *slot = match added["special"].as_bool() {
                Some(true) => TokenType::Control,
                _ => TokenType::UserDefined,
            };
        }
        types
    }

    fn parse(bytes: &[u8]) -> Value {
        serde_json::from_slice(bytes).expect("writer output is json")
    }

    fn load(directory: &str, pre_type: PreType) -> Reference {
        let plain_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(FIXTURES)
            .join(directory)
            .join("plain");
        let json_bytes =
            fs::read(plain_dir.join("tokenizer.json")).expect("vendored tokenizer.json");
        let config_bytes = fs::read(plain_dir.join("tokenizer_config.json"))
            .expect("vendored tokenizer_config.json");
        let (json, config) = (parse(&json_bytes), parse(&config_bytes));
        let [bos, eos, unk] = special_ids(&json, &config);
        let plain = vocab_from_tokenizer_json(&json_bytes, bos, eos, unk)
            .expect("reader loads fixture")
            .with_pre_type(pre_type);
        let (add_bos, add_eos) =
            bos_eos_policy_from_tokenizer_config(&config_bytes).expect("policy");
        let typed = plain
            .clone()
            .with_token_types(token_types(&json, plain.len()))
            .expect("types match vocab length")
            .with_bos_eos_policy(add_bos, add_eos);
        let written_json = parse(&tokenizer_json_bytes(&typed).expect("json written"));
        let written_config = parse(&tokenizer_config_json_bytes(&typed).expect("config written"));
        Reference {
            json,
            config,
            plain,
            typed,
            written_json,
            written_config,
        }
    }

    fn within_vocab(added_tokens: &Value, len: usize) -> Vec<Value> {
        added_tokens
            .as_array()
            .expect("added_tokens array")
            .iter()
            .filter(|added| added["id"].as_u64().is_some_and(|id| id < len as u64))
            .cloned()
            .collect()
    }

    fn granite() -> &'static Reference {
        static CELL: OnceLock<Reference> = OnceLock::new();
        CELL.get_or_init(|| load("granite-3.1-1b-a400m-moe", PreType::Gpt2))
    }

    fn gemma() -> &'static Reference {
        static CELL: OnceLock<Reference> = OnceLock::new();
        CELL.get_or_init(|| load("gemma4-e2b", PreType::Default))
    }

    fn read_back(vocab: &Vocab) -> Vocab {
        let json_bytes = tokenizer_json_bytes(vocab).expect("json written");
        let config_bytes = tokenizer_config_json_bytes(vocab).expect("config written");
        let (json, config) = (parse(&json_bytes), parse(&config_bytes));
        let [bos, eos, unk] = special_ids(&json, &config);
        let read =
            vocab_from_tokenizer_json(&json_bytes, bos, eos, unk).expect("reader loads output");
        let (add_bos, add_eos) =
            bos_eos_policy_from_tokenizer_config(&config_bytes).expect("policy");
        let typed = match json["added_tokens"].as_array().map(Vec::len) {
            Some(0) | None => read,
            Some(_) => read
                .with_token_types(token_types(&json, vocab.len()))
                .expect("types"),
        };
        typed
            .with_bos_eos_policy(add_bos, add_eos)
            .with_pre_type(vocab.pre_type())
    }

    fn renamed(vocab: &Vocab, id: u32, token: &str) -> Vocab {
        let tokens = (0..vocab.len())
            .map(|index| u32::try_from(index).expect("id fits"))
            .map(|index| {
                String::from(if index == id {
                    token
                } else {
                    vocab.token_str(index).expect("token")
                })
            })
            .collect();
        let merges: Vec<String> = vocab
            .merge_pairs()
            .into_iter()
            .map(|(left, right)| {
                alloc::format!(
                    "{} {}",
                    vocab.token_str(left).expect("left"),
                    vocab.token_str(right).expect("right")
                )
            })
            .collect();
        Vocab::new(
            tokens,
            &merges,
            vocab.bos_token_id(),
            vocab.eos_token_id(),
            vocab.unknown_token_id(),
        )
        .expect("renamed vocab builds")
        .with_pre_type(vocab.pre_type())
    }

    fn model_vocab_agrees(written: &Value, reference: &Value) -> bool {
        written["model"]["vocab"] == reference["model"]["vocab"]
    }

    fn special_strings_agree(written: &Value, reference: &Value) -> bool {
        ["bos_token", "eos_token", "unk_token"]
            .iter()
            .all(|key| written.get(key) == reference.get(key) && reference.get(key).is_some())
    }

    #[test]
    fn hf_write_output_read_by_the_hf_reader_gives_an_equal_vocab() {
        let source = &granite().plain;
        assert_eq!(&read_back(source), source);
    }

    #[test]
    fn hf_write_vocab_with_added_tokens_round_trips() {
        let source = &granite().typed;
        assert_eq!(source.token_type(0), Some(TokenType::Control));
        assert_eq!(&read_back(source), source);
    }

    #[test]
    fn hf_write_granite_model_vocab_equals_vendored() {
        let reference = granite();
        assert!(model_vocab_agrees(&reference.written_json, &reference.json));
    }

    #[test]
    fn hf_write_granite_merges_equal_vendored_in_order() {
        let reference = granite();
        assert_eq!(
            reference.written_json["model"]["merges"],
            reference.json["model"]["merges"]
        );
    }

    #[test]
    fn hf_write_granite_added_tokens_equal_vendored() {
        let reference = granite();
        let vendored = reference.json["added_tokens"]
            .as_array()
            .expect("added_tokens array");
        let in_vocab = within_vocab(&reference.json["added_tokens"], reference.typed.len());
        assert_eq!(
            reference.written_json["added_tokens"].as_array(),
            Some(&in_vocab)
        );
        assert_eq!(
            vendored.len() - in_vocab.len(),
            3,
            "ids 49152..=49154 sit beyond model.vocab"
        );
    }

    #[test]
    fn hf_write_granite_bos_eos_unk_strings_equal_vendored() {
        let reference = granite();
        assert!(special_strings_agree(
            &reference.written_config,
            &reference.config
        ));
    }

    #[test]
    fn hf_write_granite_add_bos_token_equals_vendored() {
        let reference = granite();
        assert_eq!(
            reference.written_config.get("add_bos_token"),
            Some(&Value::Bool(false))
        );
        assert_eq!(
            reference.written_config.get("add_bos_token"),
            reference.config.get("add_bos_token")
        );
    }

    #[test]
    fn hf_write_gemma4_model_vocab_equals_vendored() {
        let reference = gemma();
        assert!(model_vocab_agrees(&reference.written_json, &reference.json));
    }

    #[test]
    fn hf_write_gemma4_merges_equal_vendored_in_order() {
        let reference = gemma();
        assert_eq!(
            reference.written_json["model"]["merges"],
            reference.json["model"]["merges"]
        );
    }

    #[test]
    fn hf_write_gemma4_added_tokens_equal_vendored() {
        let reference = gemma();
        assert_eq!(
            reference.written_json["added_tokens"],
            reference.json["added_tokens"]
        );
    }

    #[test]
    fn hf_write_gemma4_bos_eos_unk_strings_equal_vendored() {
        let reference = gemma();
        assert!(special_strings_agree(
            &reference.written_config,
            &reference.config
        ));
    }

    #[test]
    fn hf_write_gemma4_add_bos_token_absent_as_in_vendored() {
        let reference = gemma();
        assert_eq!(reference.config.get("add_bos_token"), None);
        assert_eq!(reference.written_config.get("add_bos_token"), None);
    }

    #[test]
    fn hf_write_identity_control_reference_agrees_with_itself() {
        let reference = granite();
        assert!(model_vocab_agrees(&reference.json, &reference.json));
        assert!(special_strings_agree(&reference.config, &reference.config));
    }

    #[test]
    fn hf_write_degenerate_control_one_altered_vocab_entry_disagrees() {
        let reference = granite();
        let altered = renamed(&reference.typed, 0, "<|altered_end_of_text|>");
        let written = parse(&tokenizer_json_bytes(&altered).expect("json written"));
        assert!(!model_vocab_agrees(&written, &reference.json));
    }

    const PIPELINE_KEYS: [&str; 4] = ["normalizer", "pre_tokenizer", "post_processor", "decoder"];
    const MODEL_EXTRA_KEYS: [&str; 7] = [
        "dropout",
        "unk_token",
        "continuing_subword_prefix",
        "end_of_word_suffix",
        "fuse_unk",
        "byte_fallback",
        "ignore_merges",
    ];

    fn differing_keys(written: &Value, reference: &Value, keys: &[&str]) -> Vec<String> {
        keys.iter()
            .filter(|key| written.get(**key) != reference.get(**key))
            .map(|key| String::from(*key))
            .collect()
    }

    #[test]
    fn hf_write_gemma4_pipeline_blocks_equal_vendored() {
        let reference = gemma();
        assert_eq!(
            differing_keys(&reference.written_json, &reference.json, &PIPELINE_KEYS),
            Vec::<String>::new()
        );
    }

    #[test]
    fn hf_write_gemma4_model_extras_equal_vendored() {
        let reference = gemma();
        assert_eq!(
            differing_keys(
                &reference.written_json["model"],
                &reference.json["model"],
                &MODEL_EXTRA_KEYS
            ),
            Vec::<String>::new()
        );
    }

    #[test]
    fn hf_write_granite_gpt2_pipeline_blocks_equal_vendored() {
        let reference = granite();
        assert_eq!(
            differing_keys(&reference.written_json, &reference.json, &PIPELINE_KEYS),
            Vec::<String>::new()
        );
    }

    #[test]
    fn hf_write_granite_model_extras_equal_vendored() {
        let reference = granite();
        assert_eq!(
            differing_keys(
                &reference.written_json["model"],
                &reference.json["model"],
                &MODEL_EXTRA_KEYS
            ),
            Vec::<String>::new()
        );
    }

    #[test]
    fn hf_write_digit_isolated_gpt2_pre_split_is_digits_then_byte_level() {
        let isolated = granite()
            .typed
            .clone()
            .with_pre_type(PreType::DigitIsolatedGpt2);

        let written = parse(&tokenizer_json_bytes(&isolated).expect("json written"));

        let steps = written["pre_tokenizer"]["pretokenizers"]
            .as_array()
            .expect("sequence of pre-tokenizers");
        assert_eq!(steps[0]["type"], "Digits");
        assert_eq!(steps[0]["individual_digits"], true);
        assert_eq!(steps[1], granite().json["pre_tokenizer"]);
    }

    #[test]
    fn hf_write_byte_level_pre_split_without_a_hugging_face_spelling_is_refused() {
        let unspelled = granite().typed.clone().with_pre_type(PreType::Default);

        let refusal = tokenizer_json_bytes(&unspelled).expect_err("default pre-split");

        assert!(matches!(
            refusal,
            TokenizerError::HfTokenizerJsonWrite { reason } if reason.contains("Default")
        ));
    }
}
