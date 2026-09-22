use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use proxima_gguf::value::{MetadataArray, MetadataValue};

fn main() -> std::io::Result<()> {
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
        return Err(std::io::Error::other("gguf metadata region did not fit"));
    };
    let tokens = match parsed.metadata_value("tokenizer.ggml.tokens") {
        Some(MetadataValue::Array(MetadataArray::String(tokens))) => tokens,
        _ => panic!("tokens missing"),
    };
    let scores = match parsed.metadata_value("tokenizer.ggml.scores") {
        Some(MetadataValue::Array(MetadataArray::F32(scores))) => scores,
        _ => panic!("scores missing"),
    };
    let token_type = match parsed.metadata_value("tokenizer.ggml.token_type") {
        Some(MetadataValue::Array(MetadataArray::I32(token_type))) => token_type,
        _ => panic!("token_type missing"),
    };
    for id in [
        0usize, 1, 2, 3, 4, 5, 270, 5279, 41626, 245237, 260000, 262143,
    ] {
        println!(
            "id={id} token={:?} score={:?} type={:?}",
            tokens.get(id),
            scores.get(id),
            token_type.get(id)
        );
    }
    let mut nonzero = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &score in scores {
        if score != 0.0 {
            nonzero += 1;
        }
        min = min.min(score);
        max = max.max(score);
    }
    println!(
        "scores: nonzero={nonzero}/{} min={min} max={max}",
        scores.len()
    );
    Ok(())
}
