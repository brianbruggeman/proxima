// scratch diagnostic: dumps the real gemma4 checkpoint header for SLICE 1
// hparams/tensor-name verification. Not library surface.
#![allow(clippy::expect_used)]

use std::env;
use std::fs::File;

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129".to_string());
    let file = File::open(&path).expect("open gemma4 gguf");
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap gemma4 gguf");
    let file_bytes: &[u8] = &mapping;
    let parsed = proxima_gguf::parse_complete(file_bytes).expect("parse gemma4 header");

    println!("metadata entries = {}", parsed.metadata.len());
    for (key, value) in &parsed.metadata {
        if key.starts_with("gemma4.") || key == "general.architecture" {
            match value {
                proxima_gguf::value::MetadataValue::Array(array) => {
                    println!("{key} = array[{}] first10={:?}", array.len(), array);
                }
                other => println!("{key} = {other:?}"),
            }
        }
    }

    println!("total tensors = {}", parsed.tensors.len());
    let mut layer0_names: Vec<&str> = parsed
        .tensors
        .iter()
        .filter(|tensor| tensor.name.starts_with("blk.0."))
        .map(|tensor| tensor.name.as_str())
        .collect();
    layer0_names.sort_unstable();
    println!("blk.0.* tensors ({}): {:#?}", layer0_names.len(), layer0_names);

    let mut layer5_names: Vec<&str> = parsed
        .tensors
        .iter()
        .filter(|tensor| tensor.name.starts_with("blk.5."))
        .map(|tensor| tensor.name.as_str())
        .collect();
    layer5_names.sort_unstable();
    println!("blk.5.* tensors ({}): {:#?}", layer5_names.len(), layer5_names);

    let global_names: Vec<&str> = parsed
        .tensors
        .iter()
        .filter(|tensor| !tensor.name.starts_with("blk."))
        .map(|tensor| tensor.name.as_str())
        .collect();
    println!("global tensors ({}): {:#?}", global_names.len(), global_names);
}
