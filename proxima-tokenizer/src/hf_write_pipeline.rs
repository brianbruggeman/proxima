//! The `normalizer`, `pre_tokenizer`, `post_processor` and `decoder` blocks
//! and the extra `model` keys a HuggingFace `tokenizer.json` needs around a
//! [`Vocab`]'s ids and merges.
//!
//! The vocab decides, as [`crate::pipe::encode`] does: a char-level vocab
//! (`▁` spells the space, `<0xXX>` byte fallback) gets the SentencePiece-style
//! blocks of a gemma `tokenizer.json`; every other BPE vocab is byte-level
//! and its [`PreType`] picks the pre-split. A pre-split with no HuggingFace
//! spelling is an error, never a silent approximation: a `tokenizer.json`
//! whose blocks disagree with the encoder encodes the same text to other ids.

use alloc::format;
use alloc::string::String;

use serde_json::{Value, json};

use crate::error::TokenizerError;
use crate::pretokenize::PreType;
use crate::vocab::Vocab;

const SPACE_MARKER: &str = "\u{2581}";

/// The blocks of one `tokenizer.json`, each a JSON value ready to embed.
pub(crate) struct Pipeline {
    pub normalizer: Value,
    pub pre_tokenizer: Value,
    pub post_processor: Value,
    pub decoder: Value,
    pub model_extras: Value,
}

/// # Errors
///
/// [`TokenizerError::HfTokenizerJsonWrite`] for a byte-level vocab whose
/// [`PreType`] has no HuggingFace pre-tokenizer spelling here.
pub(crate) fn pipeline_of(vocab: &Vocab) -> Result<Pipeline, TokenizerError> {
    match vocab.is_char_level_bpe() {
        true => Ok(char_level(vocab)),
        false => byte_level(vocab.pre_type()),
    }
}

fn char_level(vocab: &Vocab) -> Pipeline {
    let unk = vocab
        .unknown_token_id()
        .and_then(|id| vocab.token_str(id))
        .map(String::from);
    Pipeline {
        normalizer: json!({"type": "Replace", "pattern": {"String": " "}, "content": SPACE_MARKER}),
        pre_tokenizer: json!({
            "type": "Split",
            "pattern": {"String": " "},
            "behavior": "MergedWithPrevious",
            "invert": false,
        }),
        post_processor: json!({
            "type": "TemplateProcessing",
            "single": [{"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {},
        }),
        decoder: json!({
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": SPACE_MARKER}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
            ],
        }),
        model_extras: json!({
            "dropout": null,
            "unk_token": unk,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": true,
            "byte_fallback": true,
            "ignore_merges": false,
        }),
    }
}

fn byte_level(pre_type: PreType) -> Result<Pipeline, TokenizerError> {
    let word_split = json!({
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true,
        "use_regex": true,
    });
    let pre_tokenizer = match pre_type {
        PreType::Gpt2 => word_split,
        PreType::DigitIsolatedGpt2 => json!({
            "type": "Sequence",
            "pretokenizers": [{"type": "Digits", "individual_digits": true}, word_split],
        }),
        other => {
            return Err(TokenizerError::HfTokenizerJsonWrite {
                reason: format!("pre-split {other:?} has no HuggingFace pre_tokenizer spelling"),
            });
        }
    };
    Ok(Pipeline {
        normalizer: Value::Null,
        pre_tokenizer,
        post_processor: json!({
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": false,
            "use_regex": true,
        }),
        decoder: json!({
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true,
        }),
        model_extras: json!({
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
        }),
    })
}
