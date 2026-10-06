//! The names the GGUF format itself defines for a vocab: the
//! `tokenizer.ggml.model` values that pick the encoder engine and the
//! `tokenizer.ggml.pre` values that pick a [`PreType`]. Both are file-format
//! enumerations (llama.cpp's `LLAMA_VOCAB_TYPE_*` and `LLAMA_VOCAB_PRE_TYPE_*`
//! spelled as strings), so the family and model names in them are values read
//! from a file, never a branch on which checkpoint is loaded. They live in this
//! one module so the rest of the crate selects behavior from a [`PreType`] or
//! from the arrays a file carries.

use crate::pretokenize::PreType;

/// `tokenizer.ggml.model` of a scores-driven SentencePiece vocab
/// (`LLAMA_VOCAB_TYPE_SPM`): no merges, one unigram score per token.
pub(crate) const MODEL_UNIGRAM: &str = "llama";

/// `tokenizer.ggml.model` of a merges-driven byte-level BPE vocab
/// (`LLAMA_VOCAB_TYPE_BPE`).
pub(crate) const MODEL_BYTE_LEVEL_BPE: &str = "gpt2";

impl PreType {
    /// Maps a `tokenizer.ggml.pre` value exactly as llama.cpp's
    /// `llama_vocab::impl::load` does (`llama-vocab.cpp:2168-2425`), for the
    /// pre types whose rule this scanner expresses. `None` for every other
    /// value, including ones llama.cpp itself accepts. Names whose branch also
    /// sets an encode-affecting flag this crate does not model (`add_sep` on
    /// jina-v1-en, jina-v2-code and roberta-bpe) are not mapped.
    #[must_use]
    pub fn from_gguf_name(name: &str) -> Option<Self> {
        match name {
            "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral"
            | "midm-2.0" | "lfm2" | "jina-v5-nano" | "dbrx" | "smaug-bpe" | "glm4"
            | "chatglm-bpe" => Some(Self::GroupedDigits),
            "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "megrez" | "stablelm2"
            | "hunyuan" | "solar-open" | "grok-2" => Some(Self::SingleDigit),
            "qwen35" => Some(Self::SingleDigitMarks),
            "default" => Some(Self::Default),
            "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es"
            | "jina-v2-de" | "a.x-4.0" | "mellum" | "modern-bert" | "exaone4" | "mpt"
            | "olmo" | "jais" | "trillion" | "granite-docling" => Some(Self::Gpt2),
            "starcoder" | "refact" | "command-r" | "smollm" | "codeshell" | "exaone"
            | "minerva-7b" | "mellum2" => Some(Self::DigitIsolatedGpt2),
            "falcon" => Some(Self::Falcon),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_family_names_map_to_the_pass_pipelines_like_llama_cpp() {
        assert_eq!(PreType::from_gguf_name("stablelm2"), Some(PreType::SingleDigit));
        assert_eq!(PreType::from_gguf_name("hunyuan"), Some(PreType::SingleDigit));
        assert_eq!(PreType::from_gguf_name("solar-open"), Some(PreType::SingleDigit));
        assert_eq!(PreType::from_gguf_name("dbrx"), Some(PreType::GroupedDigits));
        assert_eq!(PreType::from_gguf_name("mpt"), Some(PreType::Gpt2));
        assert_eq!(PreType::from_gguf_name("gpt-2"), Some(PreType::Gpt2));
        assert_eq!(PreType::from_gguf_name("starcoder"), Some(PreType::DigitIsolatedGpt2));
        assert_eq!(PreType::from_gguf_name("falcon"), Some(PreType::Falcon));
    }

    #[test]
    fn gguf_pre_names_map_like_llama_cpp() {
        assert_eq!(PreType::from_gguf_name("llama-bpe"), Some(PreType::GroupedDigits));
        assert_eq!(PreType::from_gguf_name("lfm2"), Some(PreType::GroupedDigits));
        assert_eq!(PreType::from_gguf_name("qwen2"), Some(PreType::SingleDigit));
        assert_eq!(PreType::from_gguf_name("deepseek-r1-qwen"), Some(PreType::SingleDigit));
        assert_eq!(PreType::from_gguf_name("megrez"), Some(PreType::SingleDigit));
        assert_eq!(PreType::from_gguf_name("qwen35"), Some(PreType::SingleDigitMarks));
        assert_eq!(PreType::from_gguf_name("deepseek-coder"), None);
        assert_eq!(PreType::from_gguf_name("default"), Some(PreType::Default));
        assert_eq!(PreType::from_gguf_name("jina-v1-en"), None);
        assert_eq!(PreType::from_gguf_name("hunyuan-dense"), None);
        assert_eq!(PreType::from_gguf_name(""), None);
    }
}
