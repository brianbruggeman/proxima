//! `draft-simple`'s vocabulary gate, matching llama.cpp's
//! `common_speculative_are_compatible` predicates.

use alloc::format;
use alloc::string::{String, ToString};

use proxima_gguf::{MetadataArray, MetadataValue, ParsedGguf};
use proxima_tokenizer::Vocab;

use crate::error::InteropError;

const TOKENIZER_MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENIZER_PRE_KEY: &str = "tokenizer.ggml.pre";
const TOKENIZER_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const BOS_ID_KEY: &str = "tokenizer.ggml.bos_token_id";
const EOS_ID_KEY: &str = "tokenizer.ggml.eos_token_id";
const ADD_BOS_KEY: &str = "tokenizer.ggml.add_bos_token";
const ADD_EOS_KEY: &str = "tokenizer.ggml.add_eos_token";
const VOCAB_MAX_SIZE_DIFFERENCE: usize = 128;
const VOCAB_CHECK_START_TOKEN_ID: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlamaVocabType {
    None,
    Spm,
    Bpe,
    Wpm,
    Ugm,
    Rwkv,
    Plamo2,
    Test,
}

impl LlamaVocabType {
    fn from_gguf(model: &str) -> Option<Self> {
        match model {
            "none" | "no_vocab" => Some(Self::None),
            "llama" => Some(Self::Spm),
            "gpt2" | "hybriddna" | "whitespace" | "gemma4" => Some(Self::Bpe),
            "bert" => Some(Self::Wpm),
            "t5" => Some(Self::Ugm),
            "rwkv" => Some(Self::Rwkv),
            "plamo2" => Some(Self::Plamo2),
            "test" => Some(Self::Test),
            _ => None,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Spm => "SPM",
            Self::Bpe => "BPE",
            Self::Wpm => "WPM",
            Self::Ugm => "UGM",
            Self::Rwkv => "RWKV",
            Self::Plamo2 => "PLAMO2",
            Self::Test => "TEST",
        }
    }

    const fn default_bos_eos(self) -> (bool, bool, Option<u32>, Option<u32>) {
        match self {
            Self::Spm => (true, false, Some(1), Some(2)),
            Self::Wpm => (true, false, Some(101), None),
            Self::Bpe => (false, false, Some(11), Some(11)),
            Self::Ugm => (false, true, None, Some(1)),
            Self::Rwkv | Self::None | Self::Test => (false, false, None, None),
            Self::Plamo2 => (false, false, Some(1), Some(2)),
        }
    }
}

/// Enforces llama.cpp's target/draft vocab compatibility checks against the
/// loaded vocabulary text and the source GGUF metadata.
pub(crate) fn ensure_compatible(
    target_metadata: &ParsedGguf,
    target_vocab: &Vocab,
    draft_metadata: &ParsedGguf,
    draft_vocab: &Vocab,
) -> Result<(), InteropError> {
    let target_type = vocab_type(target_metadata)?;
    let draft_type = vocab_type(draft_metadata)?;
    if target_type != draft_type {
        return Err(incompatible(
            "vocab type",
            target_type.name(),
            draft_type.name(),
        ));
    }

    let target_tokens = token_count(target_metadata)?;
    let draft_tokens = token_count(draft_metadata)?;
    let target_bos = effective_special_token(
        target_metadata,
        target_type,
        ADD_BOS_KEY,
        BOS_ID_KEY,
        true,
        target_tokens,
    );
    let draft_bos = effective_special_token(
        draft_metadata,
        draft_type,
        ADD_BOS_KEY,
        BOS_ID_KEY,
        true,
        draft_tokens,
    );
    if target_bos.0 != draft_bos.0 {
        return Err(incompatible(
            "add_bos",
            target_bos.0.to_string(),
            draft_bos.0.to_string(),
        ));
    }
    if target_bos.0 && target_bos.1 != draft_bos.1 {
        return Err(incompatible(
            "bos token id",
            format!("{:?}", target_bos.1),
            format!("{:?}", draft_bos.1),
        ));
    }

    let target_eos = effective_special_token(
        target_metadata,
        target_type,
        ADD_EOS_KEY,
        EOS_ID_KEY,
        false,
        target_tokens,
    );
    let draft_eos = effective_special_token(
        draft_metadata,
        draft_type,
        ADD_EOS_KEY,
        EOS_ID_KEY,
        false,
        draft_tokens,
    );
    if target_eos.0 != draft_eos.0 {
        return Err(incompatible(
            "add_eos",
            target_eos.0.to_string(),
            draft_eos.0.to_string(),
        ));
    }
    if target_eos.0 && target_eos.1 != draft_eos.1 {
        return Err(incompatible(
            "eos token id",
            format!("{:?}", target_eos.1),
            format!("{:?}", draft_eos.1),
        ));
    }

    let vocab_difference = target_tokens.abs_diff(draft_tokens);
    if vocab_difference > VOCAB_MAX_SIZE_DIFFERENCE {
        return Err(incompatible(
            "vocab size difference",
            target_tokens.to_string(),
            draft_tokens.to_string(),
        ));
    }

    for token_id in VOCAB_CHECK_START_TOKEN_ID..target_tokens.min(draft_tokens) {
        let target_text = llama_token_text(target_vocab, token_id);
        let draft_text = llama_token_text(draft_vocab, token_id);
        if target_text != draft_text {
            return Err(incompatible(
                "token text",
                format!("id {token_id}: {target_text:?}"),
                format!("id {token_id}: {draft_text:?}"),
            ));
        }
    }

    Ok(())
}

fn vocab_type(metadata: &ParsedGguf) -> Result<LlamaVocabType, InteropError> {
    let model = metadata
        .metadata_value(TOKENIZER_MODEL_KEY)
        .and_then(MetadataValue::as_str)
        .ok_or_else(|| {
            incompatible(
                "tokenizer model metadata",
                "missing or non-string",
                "required",
            )
        })?;
    LlamaVocabType::from_gguf(model).ok_or_else(|| {
        incompatible(
            "unsupported tokenizer model",
            model,
            "llama.cpp vocab type unavailable",
        )
    })
}

fn token_count(metadata: &ParsedGguf) -> Result<usize, InteropError> {
    match metadata.metadata_value(TOKENIZER_TOKENS_KEY) {
        Some(MetadataValue::Array(MetadataArray::String(tokens))) => Ok(tokens.len()),
        _ => Err(incompatible(
            "tokenizer tokens metadata",
            "missing or non-string array",
            "required",
        )),
    }
}

fn effective_special_token(
    metadata: &ParsedGguf,
    vocab_type: LlamaVocabType,
    add_key: &str,
    id_key: &str,
    is_bos: bool,
    token_count: usize,
) -> (bool, Option<u32>) {
    let (mut add_bos, mut add_eos, default_bos, default_eos) = vocab_type.default_bos_eos();
    let model = metadata
        .metadata_value(TOKENIZER_MODEL_KEY)
        .and_then(MetadataValue::as_str);
    let pre = if model == Some("gemma4") {
        Some("gemma4")
    } else {
        metadata
            .metadata_value(TOKENIZER_PRE_KEY)
            .and_then(MetadataValue::as_str)
    };
    if vocab_type == LlamaVocabType::Bpe
        && matches!(
            pre,
            Some(
                "llama3"
                    | "llama-v3"
                    | "llama-bpe"
                    | "falcon3"
                    | "falcon-h1"
                    | "pixtral"
                    | "midm-2.0"
                    | "lfm2"
                    | "jina-v5-nano"
                    | "tekken"
                    | "chameleon"
            )
        )
    {
        add_bos = true;
    }
    let default_add = if is_bos { &mut add_bos } else { &mut add_eos };
    if let Some(MetadataValue::Bool(value)) = metadata.metadata_value(add_key) {
        *default_add = *value;
    }
    if vocab_type == LlamaVocabType::Bpe
        && matches!(pre, Some("gemma4" | "granite-embed-multi-311m"))
        && is_bos
    {
        add_bos = true;
    }
    let fallback = if model == Some("gemma4")
        || (is_bos
            && vocab_type == LlamaVocabType::Bpe
            && matches!(pre, Some("glm4" | "chatglm-bpe")))
    {
        None
    } else if is_bos {
        default_bos
    } else {
        default_eos
    };
    let configured = metadata
        .metadata_value(id_key)
        .and_then(MetadataValue::as_u32)
        .filter(|token_id| (*token_id as usize) < token_count);
    let effective_id =
        configured.or_else(|| fallback.filter(|token_id| (*token_id as usize) < token_count));
    (if is_bos { add_bos } else { add_eos }, effective_id)
}

fn llama_token_text(vocab: &Vocab, token_id: usize) -> String {
    vocab
        .token_str(token_id as u32)
        .map(|token| token.split('\0').next().unwrap_or_default())
        .filter(|token| !token.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("[EMPTY_{token_id}]"))
}

fn incompatible(
    criterion: &'static str,
    target: impl ToString,
    draft: impl ToString,
) -> InteropError {
    InteropError::IncompatibleDraftModelVocab {
        criterion,
        target: target.to_string(),
        draft: draft.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    use proxima_gguf::{MetadataArray, MetadataValue, ParsedGguf};
    use proxima_tokenizer::Vocab;

    use super::ensure_compatible;

    fn vocab_tokens(count: usize) -> Vec<String> {
        let mut tokens: Vec<String> = (0..=u8::MAX)
            .map(|byte| format!("<0x{byte:02X}>"))
            .collect();
        tokens.resize_with(count, || String::from("extra"));
        for (index, token) in tokens.iter_mut().enumerate().skip(256) {
            *token = format!("extra-{index}");
        }
        tokens
    }

    fn parsed_vocab(tokens: &[String]) -> (ParsedGguf, Vocab) {
        let vocab = match Vocab::new_unigram(
            tokens.to_vec(),
            vec![0.0; tokens.len()],
            Some(1),
            Some(2),
            None,
        ) {
            Ok(vocab) => vocab,
            Err(error) => panic!("fixture unigram vocab builds: {error:?}"),
        };
        let metadata = vec![
            (
                String::from("tokenizer.ggml.model"),
                MetadataValue::String(String::from("llama")),
            ),
            (
                String::from("tokenizer.ggml.tokens"),
                MetadataValue::Array(MetadataArray::String(tokens.to_vec())),
            ),
            (
                String::from("tokenizer.ggml.bos_token_id"),
                MetadataValue::U32(1),
            ),
            (
                String::from("tokenizer.ggml.eos_token_id"),
                MetadataValue::U32(2),
            ),
        ];
        (
            ParsedGguf {
                version: 3,
                tensor_count: 0,
                kv_count: metadata.len() as u64,
                metadata,
                tensors: Vec::new(),
                data_offset: 0,
                alignment: 32,
            },
            vocab,
        )
    }

    #[test]
    fn draft_model_vocab_mismatch_is_typed_error() {
        let target_tokens = vocab_tokens(260);
        let mut draft_tokens = target_tokens.clone();
        draft_tokens.swap(5, 256);
        let (target_metadata, target_vocab) = parsed_vocab(&target_tokens);
        let (draft_metadata, draft_vocab) = parsed_vocab(&draft_tokens);

        let error = ensure_compatible(
            &target_metadata,
            &target_vocab,
            &draft_metadata,
            &draft_vocab,
        )
        .err()
        .unwrap_or_else(|| panic!("upstream checks token text at id 5"));

        assert!(matches!(
            error,
            crate::InteropError::IncompatibleDraftModelVocab {
                criterion: "token text",
                ..
            }
        ));
    }

    #[test]
    fn draft_vocab_text_before_upstream_check_start_is_not_compared() {
        let target_tokens = vocab_tokens(260);
        let mut draft_tokens = target_tokens.clone();
        draft_tokens[0] = String::from("ignored-draft-token");
        let (target_metadata, target_vocab) = parsed_vocab(&target_tokens);
        let (draft_metadata, draft_vocab) = parsed_vocab(&draft_tokens);

        ensure_compatible(
            &target_metadata,
            &target_vocab,
            &draft_metadata,
            &draft_vocab,
        )
        .unwrap_or_else(|error| panic!("upstream token-text loop starts at token id 5: {error:?}"));
    }

    #[test]
    fn draft_vocab_size_difference_at_upstream_limit_is_accepted() {
        let target_tokens = vocab_tokens(256);
        let draft_tokens = vocab_tokens(384);
        let (target_metadata, target_vocab) = parsed_vocab(&target_tokens);
        let (draft_metadata, draft_vocab) = parsed_vocab(&draft_tokens);

        ensure_compatible(
            &target_metadata,
            &target_vocab,
            &draft_metadata,
            &draft_vocab,
        )
        .unwrap_or_else(|error| panic!("difference of the upstream limit is admitted: {error:?}"));
    }

    #[test]
    fn draft_vocab_size_difference_above_upstream_limit_is_rejected() {
        let target_tokens = vocab_tokens(256);
        let draft_tokens = vocab_tokens(385);
        let (target_metadata, target_vocab) = parsed_vocab(&target_tokens);
        let (draft_metadata, draft_vocab) = parsed_vocab(&draft_tokens);

        let error = ensure_compatible(
            &target_metadata,
            &target_vocab,
            &draft_metadata,
            &draft_vocab,
        )
        .err()
        .unwrap_or_else(|| panic!("difference 129 exceeds the upstream limit 128"));

        assert!(matches!(
            error,
            crate::InteropError::IncompatibleDraftModelVocab {
                criterion: "vocab size difference",
                ..
            }
        ));
    }

    #[test]
    fn draft_vocab_bpe_special_defaults_follow_pre_tokenizer_and_overrides() {
        let (mut metadata, _) = parsed_vocab(&vocab_tokens(260));
        metadata
            .metadata
            .retain(|(key, _)| key != "tokenizer.ggml.bos_token_id");
        metadata.metadata[0].1 = MetadataValue::String(String::from("gpt2"));
        metadata.metadata.push((
            String::from("tokenizer.ggml.pre"),
            MetadataValue::String(String::from("llama3")),
        ));
        let effective = |metadata: &ParsedGguf| {
            super::effective_special_token(
                metadata,
                super::LlamaVocabType::Bpe,
                super::ADD_BOS_KEY,
                super::BOS_ID_KEY,
                true,
                260,
            )
        };
        assert_eq!(effective(&metadata), (true, Some(11)));
        metadata
            .metadata
            .push((String::from(super::ADD_BOS_KEY), MetadataValue::Bool(false)));
        assert_eq!(effective(&metadata), (false, Some(11)));
        metadata.metadata[0].1 = MetadataValue::String(String::from("gemma4"));
        assert_eq!(effective(&metadata), (true, None));
        metadata
            .metadata
            .push((String::from(super::BOS_ID_KEY), MetadataValue::U32(2)));
        assert_eq!(effective(&metadata), (true, Some(2)));
    }

    #[proxima::test]
    #[case::glm4("glm4")]
    #[case::chatglm("chatglm-bpe")]
    async fn draft_vocab_chatglm_has_no_default_bos(#[case] pre: &str) {
        let (mut target_metadata, target_vocab) = parsed_vocab(&vocab_tokens(260));
        target_metadata
            .metadata
            .retain(|(key, _)| key != crate::draft_model_compatibility::BOS_ID_KEY);
        target_metadata.metadata[0].1 = MetadataValue::String(String::from("gpt2"));
        target_metadata.metadata.push((
            String::from(crate::draft_model_compatibility::ADD_BOS_KEY),
            MetadataValue::Bool(true),
        ));
        let mut draft_metadata = target_metadata.clone();
        target_metadata.metadata.push((
            String::from(crate::draft_model_compatibility::TOKENIZER_PRE_KEY),
            MetadataValue::String(String::from(pre)),
        ));
        draft_metadata.metadata.push((
            String::from(crate::draft_model_compatibility::BOS_ID_KEY),
            MetadataValue::U32(11),
        ));

        assert!(matches!(
            ensure_compatible(
                &target_metadata,
                &target_vocab,
                &draft_metadata,
                &target_vocab,
            ),
            Err(crate::InteropError::IncompatibleDraftModelVocab {
                criterion: "bos token id",
                ..
            })
        ));

        target_metadata.metadata.push((
            String::from(crate::draft_model_compatibility::BOS_ID_KEY),
            MetadataValue::U32(11),
        ));
        ensure_compatible(
            &target_metadata,
            &target_vocab,
            &draft_metadata,
            &target_vocab,
        )
        .unwrap_or_else(|error| panic!("explicit BOS overrides chatglm null default: {error:?}"));
    }

    #[test]
    fn draft_vocab_text_uses_upstream_c_string_boundary() {
        let target_tokens = vocab_tokens(260);
        let mut draft_tokens = target_tokens.clone();
        draft_tokens[258].push_str("\0ignored-suffix");
        let (target_metadata, target_vocab) = parsed_vocab(&target_tokens);
        let (draft_metadata, draft_vocab) = parsed_vocab(&draft_tokens);
        ensure_compatible(
            &target_metadata,
            &target_vocab,
            &draft_metadata,
            &draft_vocab,
        )
        .unwrap_or_else(|error| panic!("strcmp stops at the first nul byte: {error:?}"));
    }
}
