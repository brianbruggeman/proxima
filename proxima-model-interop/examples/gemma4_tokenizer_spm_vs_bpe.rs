use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use proxima_gguf::value::{MetadataArray, MetadataValue};
use proxima_model_interop::InteropError;
use proxima_tokenizer::vocab::Vocab;

fn main() -> Result<(), InteropError> {
    let candidate = Path::new(
        "/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129",
    );
    let mut file = File::open(candidate)?;
    let mut header_buf = Vec::new();
    let mut grown = None;
    for cap in [4usize << 20, 16 << 20, 64 << 20, 256 << 20] {
        header_buf.resize(cap, 0);
        file.seek(SeekFrom::Start(0))?;
        let read = file.read(&mut header_buf)?;
        header_buf.truncate(read);
        if let Ok(parsed) = proxima_gguf::pipe::parse_complete(&header_buf) {
            grown = Some(parsed);
            break;
        }
    }
    let Some(parsed) = grown else {
        return Err(InteropError::SidecarIo(std::io::Error::other(
            "gguf metadata region did not fit",
        )));
    };

    // path A: current proxima dispatch -- exactly what vocab_from_metadata does
    let vocab_bpe_path = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)?;
    assert!(
        !vocab_bpe_path.is_unigram(),
        "current path must be merges-driven"
    );

    // path B: scores-driven SPM dispatch, using the same tokens + real scores array
    let tokens = match parsed.metadata_value("tokenizer.ggml.tokens") {
        Some(MetadataValue::Array(MetadataArray::String(tokens))) => tokens.clone(),
        _ => {
            return Err(InteropError::MissingMetadataKey {
                key: "tokenizer.ggml.tokens".to_string(),
            });
        }
    };
    let scores = match parsed.metadata_value("tokenizer.ggml.scores") {
        Some(MetadataValue::Array(MetadataArray::F32(scores))) => scores.clone(),
        _ => {
            return Err(InteropError::MissingMetadataKey {
                key: "tokenizer.ggml.scores".to_string(),
            });
        }
    };
    let bos = match parsed.metadata_value("tokenizer.ggml.bos_token_id") {
        Some(MetadataValue::U32(value)) => Some(*value),
        _ => None,
    };
    let eos = match parsed.metadata_value("tokenizer.ggml.eos_token_id") {
        Some(MetadataValue::U32(value)) => Some(*value),
        _ => None,
    };
    let unk = match parsed.metadata_value("tokenizer.ggml.unknown_token_id") {
        Some(MetadataValue::U32(value)) => Some(*value),
        _ => None,
    };
    let vocab_spm_path = Vocab::new_unigram(tokens, scores, bos, eos, unk)?;
    assert!(
        vocab_spm_path.is_unigram(),
        "spm path must be scores-driven"
    );

    let test_strings = [
        "The capital of France is Paris",
        "Hello world",
        "def foo():",
    ];

    for text in test_strings {
        let bpe_ids = proxima_tokenizer::encode(text, &vocab_bpe_path)?;
        let spm_ids = proxima_tokenizer::encode(text, &vocab_spm_path)?;
        println!("=== {text:?} ===");
        println!(
            "  current (gpt2-regex, merges-rank) ids = {bpe_ids:?}  pieces = {:?}",
            bpe_ids
                .iter()
                .map(|id| vocab_bpe_path.token_str(*id))
                .collect::<Vec<_>>()
        );
        println!(
            "  spm     (no-regex,   score-greedy) ids = {spm_ids:?}  pieces = {:?}",
            spm_ids
                .iter()
                .map(|id| vocab_spm_path.token_str(*id))
                .collect::<Vec<_>>()
        );
        println!("  ids match: {}", bpe_ids == spm_ids);
    }
    Ok(())
}
