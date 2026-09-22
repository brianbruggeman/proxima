use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use proxima_model_interop::InteropError;

fn escape_no_prefix(text: &str) -> String {
    text.chars()
        .map(|character| {
            if character == ' ' {
                '\u{2581}'
            } else {
                character
            }
        })
        .collect()
}

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

    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)?;

    let test_strings = [
        "The capital of France is Paris",
        "Hello world",
        "def foo():",
        " leading space case",
        "12345 a five digit number",
        "I'm sure you're right, we're 3.14159265 done",
        "multiple   spaces   here",
        "line1\nline2\ttabbed",
    ];

    for text in test_strings {
        let current_ids = proxima_tokenizer::encode(text, &vocab)?;
        let whole_escaped = escape_no_prefix(text);
        let wholestring_ids =
            proxima_tokenizer::bpe::encode_pretoken(whole_escaped.as_bytes(), &vocab)?;
        println!("=== {text:?} ===");
        println!(
            "  current (GPT2-regex pretokenize + rank-BPE per span) ids = {current_ids:?} pieces = {:?}",
            current_ids
                .iter()
                .map(|id| vocab.token_str(*id))
                .collect::<Vec<_>>()
        );
        println!(
            "  whole-string rank-BPE, no regex split                 ids = {wholestring_ids:?} pieces = {:?}",
            wholestring_ids
                .iter()
                .map(|id| vocab.token_str(*id))
                .collect::<Vec<_>>()
        );
        println!("  ids match: {}", current_ids == wholestring_ids);
    }
    Ok(())
}
