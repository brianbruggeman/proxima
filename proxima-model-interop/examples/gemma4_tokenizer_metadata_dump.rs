use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use proxima_gguf::value::MetadataValue;

fn main() {
    let candidate = Path::new(
        "/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129",
    );
    let mut file = File::open(candidate).expect("open gemma4 gguf");
    let mut header_buf = Vec::new();
    let parsed = 'grow: {
        for cap in [4usize << 20, 16 << 20, 64 << 20, 256 << 20] {
            header_buf.resize(cap, 0);
            file.seek(SeekFrom::Start(0)).expect("seek");
            let read = file.read(&mut header_buf).expect("read");
            header_buf.truncate(read);
            if let Ok(parsed) = proxima_gguf::pipe::parse_complete(&header_buf) {
                break 'grow parsed;
            }
        }
        panic!("gguf metadata region did not fit");
    };

    println!("metadata entries = {}", parsed.metadata.len());
    for (key, value) in &parsed.metadata {
        if !key.starts_with("tokenizer.") {
            continue;
        }
        match value {
            MetadataValue::Array(array) => {
                println!("{key} = array len={}", array.len());
            }
            other => println!("{key} = {other:?}"),
        }
    }

    // print first/last few tokens, scores, token_types, merges for spot-check
    if let Some(MetadataValue::Array(tokens)) = parsed.metadata_value("tokenizer.ggml.tokens") {
        println!("tokens.len() = {}", tokens.len());
    }
    if let Some(value) = parsed.metadata_value("tokenizer.ggml.merges") {
        if let MetadataValue::Array(merges) = value {
            println!("merges.len() = {}", merges.len());
        }
    } else {
        println!("no tokenizer.ggml.merges key present");
    }
    if let Some(value) = parsed.metadata_value("tokenizer.ggml.scores") {
        if let MetadataValue::Array(scores) = value {
            println!("scores.len() = {}", scores.len());
        }
    } else {
        println!("no tokenizer.ggml.scores key present");
    }
    if let Some(value) = parsed.metadata_value("tokenizer.ggml.token_type") {
        if let MetadataValue::Array(token_types) = value {
            println!("token_type.len() = {}", token_types.len());
        }
    } else {
        println!("no tokenizer.ggml.token_type key present");
    }

    // spot-check the four bespoke turn/channel control ids the chat template
    // relies on, plus scan the whole vocab for their literal strings in case
    // the ids the task named do not match this exact checkpoint's vocab.
    if let (
        Some(MetadataValue::Array(proxima_gguf::value::MetadataArray::String(tokens))),
        Some(MetadataValue::Array(proxima_gguf::value::MetadataArray::I32(token_types))),
    ) = (
        parsed.metadata_value("tokenizer.ggml.tokens"),
        parsed.metadata_value("tokenizer.ggml.token_type"),
    ) {
        for id in [2u32, 100, 101, 105, 106] {
            let token = tokens.get(id as usize);
            let token_type = token_types.get(id as usize);
            println!("id={id} token={token:?} token_type={token_type:?}");
        }
        for wanted in ["<|turn>", "<turn|>", "<|channel>", "<channel|>", "<bos>"] {
            let found: Vec<(usize, &str)> = tokens
                .iter()
                .enumerate()
                .filter(|(_, text)| text.as_str() == wanted)
                .map(|(index, text)| (index, text.as_str()))
                .collect();
            println!("literal {wanted:?} -> {found:?}");
        }
    }
}
