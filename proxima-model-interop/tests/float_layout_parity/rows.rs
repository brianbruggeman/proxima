use std::path::Path;

use proxima_gguf::GgmlType;
use proxima_gguf::quant::q8_0;

const GRANITE_ENV: &str = "PROXIMA_ARCH_GRANITE_MOE_GGUF";
pub const ROWS: usize = 1024;
pub const COLUMNS: usize = 512;

pub fn granite_attention_rows() -> Vec<f32> {
    let path = std::env::var(GRANITE_ENV).unwrap_or_else(|_| {
        panic!("no granite gguf: set {GRANITE_ENV} to the granite 3.1 1b a400m blob path")
    });
    assert!(
        Path::new(&path).exists(),
        "no granite gguf at {path}: set {GRANITE_ENV}"
    );
    let (parsed, bytes) =
        proxima_gguf::edge::read_file(Path::new(&path)).expect("granite blob must parse as gguf");
    let tensor = parsed
        .tensors
        .iter()
        .find(|tensor| {
            tensor.ggml_type == GgmlType::Q8_0
                && tensor.dims.as_slice() == [ROWS as u64, COLUMNS as u64]
        })
        .expect("granite holds a 1024x512 q8_0 tensor");
    let range = parsed
        .tensor_data_range(tensor, bytes.len() as u64)
        .expect("tensor range must fit the file");
    let mut weights = vec![0.0f32; tensor.element_count() as usize];
    q8_0::dequantize(
        &bytes[range.start as usize..range.end as usize],
        &mut weights,
    )
    .expect("well-formed q8_0 tensor");
    weights
}
