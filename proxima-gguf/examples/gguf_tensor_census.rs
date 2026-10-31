//! Prints general.architecture and every tensor's ggml type, shape, element
//! count and byte length straight from a gguf directory.
//!
//! Usage: `cargo run --example gguf_tensor_census -- <path.gguf>`

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;

use proxima_gguf::pipe::parse_complete;

const METADATA_VALUE_CHARS: usize = 400;
const HEADER_CAPS: [usize; 4] = [4 << 20, 16 << 20, 64 << 20, 256 << 20];

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args().nth(1).ok_or("usage: gguf_tensor_census <path.gguf>")?;
    let mut buffer = Vec::new();
    let mut last_error = None;
    for cap in HEADER_CAPS {
        buffer.clear();
        File::open(&path)?.take(cap as u64).read_to_end(&mut buffer)?;
        match parse_complete(&buffer) {
            Ok(parsed) => return report(&path, &parsed),
            Err(error) => last_error = Some(error),
        }
    }
    Err(format!("{path}: {last_error:?}").into())
}

fn report(path: &str, parsed: &proxima_gguf::pipe::ParsedGguf) -> Result<(), Box<dyn Error>> {
    println!(
        "file={path} version={} tensors={} data_offset={}",
        parsed.version, parsed.tensor_count, parsed.data_offset
    );
    for (key, value) in &parsed.metadata {
        let rendered = format!("{value:?}");
        let shown: String = rendered.chars().take(METADATA_VALUE_CHARS).collect();
        println!("{key}={shown}");
    }
    for tensor in &parsed.tensors {
        let layout = tensor.ggml_type.block_layout();
        println!(
            "tensor name={} type={:?} wire={} dims={:?} elements={} nbytes={:?} block_elements={} block_bytes={}",
            tensor.name,
            tensor.ggml_type,
            tensor.ggml_type.to_wire(),
            tensor.dims,
            tensor.element_count(),
            tensor.nbytes(),
            layout.block_elements,
            layout.block_bytes,
        );
    }
    Ok(())
}
