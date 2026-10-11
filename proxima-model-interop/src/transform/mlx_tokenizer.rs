//! The two tokenizer files of an MLX export directory.
//!
//! A GGUF source builds its [`proxima_tokenizer::Vocab`] with
//! [`proxima_tokenizer::gguf::vocab_from_metadata`] and writes it through
//! [`proxima_tokenizer::hf_write`]; an HF-directory source carries its two
//! files across unchanged, as [`super::copy_config`] does for `config.json`.
//! The crate has no GGUF-or-directory source type, so the two entry points are
//! two functions over what each source already is: a parsed GGUF, or bytes.

use alloc::string::ToString;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_tokenizer::TokenizerError;
use proxima_tokenizer::gguf::vocab_from_metadata;
use proxima_tokenizer::hf_write::{tokenizer_config_json_bytes, tokenizer_json_bytes};
use serde_json::Value;

use crate::error::InteropError;

/// The bytes of an export's `tokenizer.json` and `tokenizer_config.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerFiles {
    pub tokenizer_json: Vec<u8>,
    pub tokenizer_config_json: Vec<u8>,
}

/// Writes the tokenizer files of a GGUF source from its tokenizer metadata.
/// Only what a `Vocab` holds is written; see `proxima_tokenizer::hf_write`
/// for the pipeline blocks it cannot reproduce.
///
/// # Errors
///
/// [`InteropError::Tokenizer`] when the metadata holds no complete vocab or
/// the vocab cannot be written.
#[must_use = "the files are the export's tokenizer"]
pub fn export_tokenizer(parsed: &ParsedGguf) -> Result<TokenizerFiles, InteropError> {
    let vocab = vocab_from_metadata(parsed)?;
    Ok(TokenizerFiles {
        tokenizer_json: tokenizer_json_bytes(&vocab)?,
        tokenizer_config_json: tokenizer_config_json_bytes(&vocab)?,
    })
}

/// Carries an HF-directory source's two tokenizer files into the export
/// unchanged. Each is checked to be a JSON object, nothing more.
///
/// # Errors
///
/// [`InteropError::Tokenizer`] when either file is not a JSON object.
#[must_use = "the files are the export's tokenizer"]
pub fn copy_tokenizer(
    tokenizer_json: &[u8],
    tokenizer_config_json: &[u8],
) -> Result<TokenizerFiles, InteropError> {
    Ok(TokenizerFiles {
        tokenizer_json: copy_object(tokenizer_json, "tokenizer.json")?,
        tokenizer_config_json: copy_object(tokenizer_config_json, "tokenizer_config.json")?,
    })
}

fn copy_object(source: &[u8], file: &str) -> Result<Vec<u8>, InteropError> {
    match serde_json::from_slice::<Value>(source) {
        Ok(Value::Object(_)) => Ok(source.to_vec()),
        Ok(_) => Err(not_an_object(file, "is not a json object")),
        Err(error) => Err(not_an_object(file, &error.to_string())),
    }
}

fn not_an_object(file: &str, reason: &str) -> InteropError {
    InteropError::Tokenizer(TokenizerError::MalformedHfTokenizerJson {
        reason: alloc::format!("{file} {reason}"),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::fs::{self, File};
    use std::path::Path;

    use proxima_gguf::parse_complete;
    use proxima_tokenizer::encode_with_bos_eos;
    use proxima_tokenizer::hf::{bos_eos_policy_from_tokenizer_config, vocab_from_tokenizer_json};

    use super::*;

    const E2B_GGUF: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
    const GRANITE_GGUF: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";
    const GRANITE: &str = "granite-3.1-1b-a400m-moe";
    const E2B: &str = "gemma4-e2b";
    const PROMPT: &str = "The capital of France is";

    fn fixtures() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn vendored(model: &str) -> TokenizerFiles {
        let plain = fixtures().join("mlx-reference").join(model).join("plain");
        TokenizerFiles {
            tokenizer_json: fs::read(plain.join("tokenizer.json"))
                .expect("vendored tokenizer.json"),
            tokenizer_config_json: fs::read(plain.join("tokenizer_config.json"))
                .expect("vendored tokenizer_config.json"),
        }
    }

    fn written(gguf: &str) -> TokenizerFiles {
        let file = File::open(gguf).expect("host-local gguf blob present");
        let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("gguf maps");
        export_tokenizer(&parse_complete(&mapping).expect("gguf parses")).expect("tokenizer writes")
    }

    fn parse(bytes: &[u8]) -> Value {
        serde_json::from_slice(bytes).expect("valid json")
    }

    fn added_contents(json: &Value) -> Vec<(u64, String)> {
        json["added_tokens"]
            .as_array()
            .expect("added_tokens array")
            .iter()
            .map(|added| {
                let content = added["content"].as_str().expect("content").to_string();
                (added["id"].as_u64().expect("id"), content)
            })
            .collect()
    }

    fn added_flags(json: &Value) -> Vec<(Value, Value)> {
        json["added_tokens"]
            .as_array()
            .expect("added_tokens array")
            .iter()
            .map(|added| (added["special"].clone(), added["normalized"].clone()))
            .collect()
    }

    fn vocab_agrees(written: &Value, reference: &Value) -> bool {
        let (written_vocab, reference_vocab) = (
            written["model"]["vocab"].as_object().expect("vocab"),
            reference["model"]["vocab"].as_object().expect("vocab"),
        );
        let named_added: Vec<String> = added_contents(reference)
            .into_iter()
            .map(|(_, content)| content)
            .collect();
        reference_vocab
            .iter()
            .all(|(token, id)| written_vocab.get(token) == Some(id))
            && written_vocab
                .keys()
                .filter(|token| !reference_vocab.contains_key(*token))
                .all(|token| named_added.contains(token))
    }

    fn differing_fields(left: &TokenizerFiles, right: &TokenizerFiles) -> Vec<&'static str> {
        let (left_json, right_json) = (parse(&left.tokenizer_json), parse(&right.tokenizer_json));
        let (left_config, right_config) = (
            parse(&left.tokenizer_config_json),
            parse(&right.tokenizer_config_json),
        );
        let mut differing = Vec::new();
        let mut note = |name: &'static str, same: bool| {
            if !same {
                differing.push(name);
            }
        };
        note("model.vocab", vocab_agrees(&left_json, &right_json));
        note(
            "model.merges",
            left_json["model"]["merges"] == right_json["model"]["merges"],
        );
        note(
            "added_tokens",
            added_contents(&left_json) == added_contents(&right_json),
        );
        note(
            "added_tokens.flags",
            added_flags(&left_json) == added_flags(&right_json),
        );
        for key in ["bos_token", "eos_token", "unk_token", "add_bos_token"] {
            note(key, left_config.get(key) == right_config.get(key));
        }
        differing
    }

    fn special_id(files: &TokenizerFiles, key: &str) -> Option<u32> {
        let token = parse(&files.tokenizer_config_json)
            .get(key)?
            .as_str()?
            .to_string();
        let id = parse(&files.tokenizer_json)["model"]["vocab"]
            .get(token)?
            .as_u64()?;
        u32::try_from(id).ok()
    }

    fn encode_prompt(files: &TokenizerFiles) -> Vec<u32> {
        let ids = ["bos_token", "eos_token", "unk_token"].map(|key| special_id(files, key));
        let vocab = vocab_from_tokenizer_json(&files.tokenizer_json, ids[0], ids[1], ids[2])
            .expect("written tokenizer.json reads back");
        let (add_bos, add_eos) = bos_eos_policy_from_tokenizer_config(&files.tokenizer_config_json)
            .expect("written config reads back");
        let vocab = vocab.with_bos_eos_policy(add_bos, add_eos);
        let wants_bos = vocab
            .add_bos_token()
            .unwrap_or_else(|| vocab.bos_token_id().is_some());
        encode_with_bos_eos(
            PROMPT,
            &vocab,
            wants_bos,
            vocab.add_eos_token().unwrap_or(false),
        )
        .expect("prompt encodes")
    }

    fn recorded_prompt_ids(model_directory: &str) -> Vec<u32> {
        let path = fixtures()
            .join("llama-parity")
            .join(model_directory)
            .join("llama_ids.json");
        let records = parse(&fs::read(path).expect("llama ids fixture"));
        records[0]["prompt_ids"]
            .as_array()
            .expect("prompt_ids array")
            .iter()
            .map(|id| u32::try_from(id.as_u64().expect("id is u64")).expect("id fits"))
            .collect()
    }

    fn first_difference(left: &[u8], right: &[u8]) -> Option<usize> {
        let common = left
            .iter()
            .zip(right)
            .position(|(a_byte, b_byte)| a_byte != b_byte);
        common.or((left.len() != right.len()).then_some(left.len().min(right.len())))
    }

    #[test]
    fn mlx_tokenizer_granite_written_from_gguf_equals_vendored_with_added_tokens_in_model_vocab() {
        let differing = differing_fields(&written(GRANITE_GGUF), &vendored(GRANITE));

        assert_eq!(differing, Vec::<&str>::new());
    }

    #[test]
    fn mlx_tokenizer_e2b_written_from_gguf_equals_vendored_except_gguf_only_facts() {
        let differing = differing_fields(&written(E2B_GGUF), &vendored(E2B));

        assert_eq!(differing, vec!["added_tokens.flags", "add_bos_token"]);
    }

    #[test]
    fn mlx_tokenizer_identity_control_vendored_compared_to_itself_has_no_difference() {
        let reference = vendored(GRANITE);

        assert_eq!(
            differing_fields(&reference, &reference.clone()),
            Vec::<&str>::new()
        );
    }

    #[test]
    fn mlx_tokenizer_degenerate_control_one_altered_vocab_entry_is_reported() {
        let reference = vendored(GRANITE);
        let mut altered = reference.clone();
        let mut json = parse(&altered.tokenizer_json);
        let entry = json["model"]["vocab"]
            .as_object_mut()
            .expect("vocab object");
        let moved = entry.remove("Ġthe").expect("granite vocab has the token");
        entry.insert("Ġthe_altered".to_string(), moved);
        altered.tokenizer_json = serde_json::to_vec(&json).expect("reserializes");

        assert_eq!(differing_fields(&altered, &reference), vec!["model.vocab"]);
    }

    #[test]
    fn mlx_tokenizer_granite_written_encodes_the_fixture_prompt_to_the_llama_ids() {
        let ids = encode_prompt(&written(GRANITE_GGUF));

        assert_eq!(ids, recorded_prompt_ids("granite_moe"));
    }

    #[test]
    fn mlx_tokenizer_e2b_written_encodes_the_fixture_prompt_to_the_llama_ids() {
        let ids = encode_prompt(&written(E2B_GGUF));

        assert_eq!(ids, recorded_prompt_ids("gemma4_e2b"));
    }

    #[test]
    fn mlx_tokenizer_ids_identity_control_recorded_ids_equal_themselves() {
        let recorded = recorded_prompt_ids("granite_moe");

        assert_eq!(recorded, recorded.clone());
    }

    #[test]
    fn mlx_tokenizer_ids_degenerate_control_one_altered_token_differs() {
        let recorded = recorded_prompt_ids("gemma4_e2b");
        let mut altered = recorded.clone();
        altered[2] += 1;

        assert_ne!(altered, recorded);
    }

    #[test]
    fn mlx_tokenizer_json_copy_of_an_hf_directory_source_is_byte_equal() {
        for model in [GRANITE, E2B] {
            let source = vendored(model);

            let copied = copy_tokenizer(&source.tokenizer_json, &source.tokenizer_config_json)
                .expect("vendored files are json objects");

            assert_eq!(
                first_difference(&copied.tokenizer_json, &source.tokenizer_json),
                None,
                "{model}"
            );
        }
    }

    #[test]
    fn mlx_tokenizer_json_copy_identity_control_equal_buffers_have_no_difference() {
        let source = vendored(GRANITE).tokenizer_json;

        assert_eq!(first_difference(&source, &source.clone()), None);
    }

    #[test]
    fn mlx_tokenizer_json_copy_degenerate_control_one_byte_different_source_is_located() {
        let source = vendored(GRANITE);
        let position = source
            .tokenizer_json
            .iter()
            .position(|byte| *byte == b'1')
            .expect("a digit");
        let mut altered = source.tokenizer_json.clone();
        altered[position] = b'2';

        let copied = copy_tokenizer(&altered, &source.tokenizer_config_json)
            .expect("altered file is still a json object");

        assert_eq!(
            first_difference(&copied.tokenizer_json, &source.tokenizer_json),
            Some(position)
        );
    }

    #[test]
    fn mlx_tokenizer_config_copy_of_an_hf_directory_source_is_byte_equal() {
        for model in [GRANITE, E2B] {
            let source = vendored(model);

            let copied = copy_tokenizer(&source.tokenizer_json, &source.tokenizer_config_json)
                .expect("vendored files are json objects");

            assert_eq!(
                first_difference(&copied.tokenizer_config_json, &source.tokenizer_config_json),
                None,
                "{model}"
            );
        }
    }

    #[test]
    fn mlx_tokenizer_config_copy_identity_control_equal_buffers_have_no_difference() {
        let source = vendored(E2B).tokenizer_config_json;

        assert_eq!(first_difference(&source, &source.clone()), None);
    }

    #[test]
    fn mlx_tokenizer_config_copy_degenerate_control_one_byte_different_source_is_located() {
        let source = vendored(GRANITE);
        let position = source
            .tokenizer_config_json
            .iter()
            .position(|byte| *byte == b'1')
            .expect("a digit");
        let mut altered = source.tokenizer_config_json.clone();
        altered[position] = b'2';

        let copied = copy_tokenizer(&source.tokenizer_json, &altered)
            .expect("altered file is still a json object");

        assert_eq!(
            first_difference(&copied.tokenizer_config_json, &source.tokenizer_config_json),
            Some(position)
        );
    }
}
