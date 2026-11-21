//! Bind-time decode of a codec with no packed CPU kernel, at llama's serving
//! configuration.
//!
//! `PROXIMAOMEGA-71`: `push_borrowed_part` used to refuse a Q4_1 part with
//! `UnrepresentableGgmlType` because `as_block` had no packed block for it,
//! even though `proxima_gguf::quant::dispatch::dequantize` decodes the format.
//! This bench drives the real binder over a Q4_1 fused tensor cut into three
//! `TensorAlias::Part` leaves and reports the one-time decode cost (the path a
//! checkpoint pays instead of failing the load).
//!
//! Configuration is `ServingSettings` (conflaguration), the same object
//! `decode_gbps_baseline` and `kv_scoreboard` lower, and the run asserts the
//! defaults that mirror llama-server's `-b 2048 -ub 512`. The format and the
//! payload are `proxima_numeric`'s: `NumericFormat::Q4_1` and `encode_blocks`.
//! No q4_1-specific environment variable or serving branch exists here.
//!
//! # Run
//!
//! ```sh
//! git wt cargo bench -p proxima-model-interop --features std \
//!     --bench codec_decode_fallback -- --test
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Instant;

use conflaguration::Settings;
use proxima_gguf::types::GgmlType;
use proxima_gguf::{GgufModel, TensorPayload, parse_complete, write_complete};
use proxima_model_interop::ServingSettings;
use proxima_model_interop::bind_program_leaves;
use proxima_model_interop::profiles::{BindingProfile, TensorAlias, binding_profile};
use proxima_numeric::{DType, NumericFormat, decode_blocks, encode_blocks};
use proxima_tensor::op::{Extent, Op};

/// One projection row length, a llama-scale hidden width.
const K: usize = 4096;
/// Rows per part; three parts make the fused tensor.
const ROWS: usize = 512;
const PARTS: usize = 3;

fn main() {
    let quick = std::env::args().any(|argument| argument == "--test");
    let settings = ServingSettings::from_env().expect("ServingSettings from the environment");
    assert_eq!(settings.batch_size, 2048, "llama-server's -b 2048");
    assert_eq!(settings.ubatch_size, 512, "llama-server's -ub 512");

    let format = NumericFormat::Q4_1;
    let values: Vec<f32> = (0..ROWS * PARTS * K)
        .map(|index| ((index % 37) as f32 - 18.0) / 7.0)
        .collect();
    let mut fused = vec![0u8; format.packed_len(values.len()).expect("whole Q4_1 blocks")];
    encode_blocks(&format, &values, &mut fused).expect("proxima-numeric encodes Q4_1");

    let model = GgufModel {
        version: 3,
        metadata: Vec::new(),
        tensors: vec![TensorPayload {
            name: "blk.0.shortconv.in_proj.weight".to_string(),
            dims: [K as u64, (ROWS * PARTS) as u64].into_iter().collect(),
            ggml_type: GgmlType::Q4_1,
            data: &fused,
        }],
    };
    let bytes = write_complete(&model).expect("writes the in-memory Q4_1 gguf");
    let parsed = parse_complete(&bytes).expect("parses the written gguf");

    let program: Vec<Op> = ["b", "c", "x"]
        .into_iter()
        .map(|suffix| Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(K as u32), Extent::Static(ROWS as u32)],
            name: Some(format!("blk.0.shortconv.in_proj.weight.{suffix}")),
        })
        .collect();
    let part = |suffix: &str, index: u32| TensorAlias::Part {
        leaf: format!("shortconv.in_proj.weight.{suffix}"),
        from: "shortconv.in_proj.weight".to_string(),
        part: index,
        of: PARTS as u32,
    };
    let profile = BindingProfile {
        aliases: vec![part("b", 0), part("c", 1), part("x", 2)],
        ..binding_profile("no-family-file").expect("the default binding profile parses")
    };

    let runs = if quick { 1 } else { 5 };
    let part_bytes = ROWS
        * (K / usize::from(format.elements_per_block()))
        * usize::from(format.bytes_per_block());
    let mut best = f64::INFINITY;
    let mut mismatch = 0;
    let mut resident = 0;
    for _ in 0..runs {
        let start = Instant::now();
        let weights = bind_program_leaves(&parsed, &bytes, &program, &profile, &[])
            .expect("a decodable codec binds without a packed kernel");
        best = best.min(start.elapsed().as_secs_f64());
        resident = weights.resident_bytes();
        let owned = weights.owned();
        assert_eq!(owned.len(), PARTS, "one owned buffer per part");
        for (index, (_, decoded)) in owned.iter().enumerate() {
            let first = index * part_bytes;
            let mut expected = vec![0.0f32; ROWS * K];
            decode_blocks(
                &NumericFormat::Q4_1,
                &fused[first..first + part_bytes],
                &mut expected,
            )
            .expect("the numeric crate's walker reads the same bytes");
            if decoded != &expected {
                mismatch += 1;
            }
        }
    }

    let gbps = fused.len() as f64 / best / 1e9;
    println!(
        "codec_decode_fallback format=q4_1 rows={ROWS} k={K} parts={PARTS} \
q4_1_bytes={} f32_bytes={resident} batch={} ubatch={} kv={} mismatch={mismatch} \
bind_ms={:.3} gbps={gbps:.3}",
        fused.len(),
        settings.batch_size,
        settings.ubatch_size,
        settings.kv_cache_key_quant.name().unwrap_or("unnamed"),
        best * 1e3,
    );
    assert_eq!(mismatch, 0, "every part equals the format-keyed decoder");
}
