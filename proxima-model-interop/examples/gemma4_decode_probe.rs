use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

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

    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("builds vocab via the current gemma4 dispatch");

    let text = "The capital of France is Paris";
    let ids = proxima_tokenizer::encode(text, &vocab).expect("encode");
    let decoded = proxima_tokenizer::decode(&ids, &vocab).expect("decode");
    println!("ids = {ids:?}");
    println!("decoded = {decoded:?}");
    println!(
        "contains literal U+2581 (▁): {}",
        decoded.contains('\u{2581}')
    );
    println!(
        "is_unigram (gate that skips unescape): {}",
        vocab.is_unigram()
    );
}
