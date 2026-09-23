use std::fs::File;
use memmap2::MmapOptions;
use proxima_gguf::parse_complete;

fn main() {
    let model_path = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
    let file = File::open(model_path).expect("open file");
    let mmap = unsafe { MmapOptions::new().map(&file) }.expect("mmap");
    let parsed = parse_complete(&mmap).expect("parse");
    
    for tensor in &parsed.tensors {
        let dims_str = tensor.dims.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let codec = format!("{:?}", tensor.ggml_type);
        let element_count = tensor.element_count();
        println!("{}\t[{}]\t{}\t{}", tensor.name, dims_str, codec, element_count);
    }
}
