#![allow(clippy::expect_used)]
//! Per-tensor dequantization fidelity against ggml's own decoder.
//!
//! The oracle is `ggml_get_type_traits(type)->to_float`, which for the
//! K-quants IS `dequantize_row_q4_K` / `_q5_K` / `_q6_K`
//! (`ggml/src/ggml.c:691,699,707`). Calling the trait directly means no
//! graph, no context, no threading -- the same function `ggml_get_rows`
//! and the CPU matmul path reach for, fed the exact bytes our parser
//! hands `proxima_gguf::quant`.
//!
//! Every tensor in the file, every block, is compared. Findings are
//! emitted as structured `debug!`/`warn!` records into a file-sink
//! `Exporter` (never a hand-rolled dump).

use std::env;
use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::raw::c_int;
use std::path::PathBuf;

use proxima_gguf::pipe::parse_complete;
use proxima_gguf::quant::{q4_k, q5_k, q6_k, q8_0};
use proxima_gguf::types::GgmlType;
use proxima_telemetry::export::{Exporter, Formatter};
use proxima_telemetry::level::Level;
use proxima_telemetry::recorder::Recorder;

type GgmlToFloat = unsafe extern "C" fn(*const c_void, *mut f32, i64);

#[repr(C)]
struct GgmlTypeTraits {
    type_name: *const i8,
    block_size: i64,
    block_size_interleave: i64,
    type_size: usize,
    is_quantized: bool,
    to_float: Option<GgmlToFloat>,
    from_float_ref: *const c_void,
}

#[link(name = "c++")]
unsafe extern "C" {}

#[link(name = "ggml-base", kind = "static")]
unsafe extern "C" {
    fn ggml_get_type_traits(type_: c_int) -> *const GgmlTypeTraits;
}

/// One tensor's comparison against ggml's decoder, sorted worst-first for
/// the report table.
struct FidelityRow {
    tensor_name: String,
    dtype: String,
    block_count: usize,
    max_abs_diff: f32,
    worst_block: usize,
    worst_index: usize,
    ours: f32,
    ggml: f32,
}

fn ggml_type_code(ggml_type: GgmlType) -> Option<c_int> {
    match ggml_type {
        GgmlType::F32 => Some(0),
        GgmlType::Q8_0 => Some(8),
        GgmlType::Q4_K => Some(12),
        GgmlType::Q5_K => Some(13),
        GgmlType::Q6_K => Some(14),
        _ => None,
    }
}

fn block_elements(ggml_type: GgmlType) -> usize {
    match ggml_type {
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => 256,
        GgmlType::Q8_0 => 32,
        _ => 1,
    }
}

fn ours(ggml_type: GgmlType, data: &[u8], out: &mut [f32]) -> Result<(), String> {
    match ggml_type {
        GgmlType::F32 => {
            for (slot, chunk) in out.iter_mut().zip(data.chunks_exact(4)) {
                *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(())
        }
        GgmlType::Q4_K => q4_k::dequantize(data, out).map_err(|e| e.to_string()),
        GgmlType::Q5_K => q5_k::dequantize(data, out).map_err(|e| e.to_string()),
        GgmlType::Q6_K => q6_k::dequantize(data, out).map_err(|e| e.to_string()),
        GgmlType::Q8_0 => q8_0::dequantize(data, out).map_err(|e| e.to_string()),
        other => Err(format!("no codec for {other:?}")),
    }
}

fn main() {
    let model = env::args().nth(1).map(PathBuf::from).expect("usage: <model.gguf> [log-path]");
    let log_path = env::args().nth(2).unwrap_or_else(|| "fidelity.jsonl".to_string());

    let recorder = Recorder::builder()
        .export(Exporter::file(&log_path).format(Formatter::Text))
        .expect("file exporter")
        .install()
        .expect("recorder");

    let mut file = File::open(&model).expect("open model");
    let file_len = file.metadata().expect("stat").len();

    let mut prefix = vec![0u8; (32 << 20).min(file_len as usize)];
    file.read_exact(&mut prefix).expect("read prefix");
    let parsed = parse_complete(&prefix).expect("parse gguf metadata");

    println!("tensors={} data_offset={}", parsed.tensors.len(), parsed.data_offset);

    let mut compared_tensors = 0usize;
    let mut compared_blocks = 0usize;
    let mut skipped: Vec<(String, String)> = Vec::new();
    let mut rows: Vec<FidelityRow> = Vec::new();

    for tensor in &parsed.tensors {
        let range = parsed.tensor_data_range(tensor, file_len).expect("range");
        let byte_len = (range.end - range.start) as usize;
        let element_count = tensor.element_count() as usize;

        let Some(code) = ggml_type_code(tensor.ggml_type) else {
            skipped.push((tensor.name.clone(), format!("{:?} unmapped", tensor.ggml_type)));
            continue;
        };
        let traits = unsafe { &*ggml_get_type_traits(code) };
        let Some(to_float) = traits.to_float else {
            skipped.push((tensor.name.clone(), format!("{:?} has no ggml to_float", tensor.ggml_type)));
            continue;
        };

        let mut data = vec![0u8; byte_len];
        file.seek(SeekFrom::Start(range.start)).expect("seek");
        file.read_exact(&mut data).expect("read tensor");

        let mut mine = vec![0f32; element_count];
        if let Err(reason) = ours(tensor.ggml_type, &data, &mut mine) {
            skipped.push((tensor.name.clone(), reason));
            continue;
        }

        let mut theirs = vec![0f32; element_count];
        unsafe { to_float(data.as_ptr().cast(), theirs.as_mut_ptr(), element_count as i64) };

        let mut max_diff = 0f32;
        let mut worst = 0usize;
        for index in 0..element_count {
            let diff = (mine[index] - theirs[index]).abs();
            if diff > max_diff {
                max_diff = diff;
                worst = index;
            }
        }

        let elements_per_block = block_elements(tensor.ggml_type);
        let block_count = element_count / elements_per_block;
        let worst_block = worst / elements_per_block;
        compared_tensors += 1;
        compared_blocks += block_count;

        let dtype = format!("{:?}", tensor.ggml_type);
        // this diagnostic tool runs once per tensor (226 max on a real
        // checkpoint) and needs `&'static str` tag values; leaking the
        // per-tensor name/dtype strings is bounded and cheaper than
        // threading an owned-string tag type through the recorder for a
        // one-shot comparison run.
        let tensor_name: &'static str = Box::leak(tensor.name.clone().into_boxed_str());
        let dtype_name: &'static str = Box::leak(dtype.clone().into_boxed_str());
        if max_diff > 0.0 {
            recorder
                .log()
                .level(Level::WARN)
                .message("dequant diverges from ggml")
                .module_path(module_path!())
                .tag("tensor", tensor_name)
                .tag("dtype", dtype_name)
                .tag("blocks", block_count as u64)
                .tag("max_abs_diff", f64::from(max_diff))
                .tag("worst_block", worst_block as u64)
                .tag("worst_index", worst as u64)
                .tag("ours", f64::from(mine[worst]))
                .tag("ggml", f64::from(theirs[worst]))
                .emit();
        } else {
            recorder
                .log()
                .level(Level::DEBUG)
                .message("dequant matches ggml on every block")
                .module_path(module_path!())
                .tag("tensor", tensor_name)
                .tag("dtype", dtype_name)
                .tag("blocks", block_count as u64)
                .tag("max_abs_diff", 0.0f64)
                .emit();
        }
        rows.push(FidelityRow {
            tensor_name: tensor.name.clone(),
            dtype,
            block_count,
            max_abs_diff: max_diff,
            worst_block,
            worst_index: worst,
            ours: mine[worst],
            ggml: theirs[worst],
        });
    }

    while recorder.drain() > 0 {}

    rows.sort_by(|a, b| {
        b.max_abs_diff.total_cmp(&a.max_abs_diff).then_with(|| a.tensor_name.cmp(&b.tensor_name))
    });
    println!("\nname dtype blocks max_abs_diff worst_block worst_index ours ggml");
    for row in &rows {
        println!(
            "{} {} {} {:e} {} {} {:e} {:e}",
            row.tensor_name,
            row.dtype,
            row.block_count,
            row.max_abs_diff,
            row.worst_block,
            row.worst_index,
            row.ours,
            row.ggml
        );
    }
    println!(
        "\ncompared_tensors={compared_tensors}/{} compared_blocks={compared_blocks} skipped={}",
        parsed.tensors.len(),
        skipped.len()
    );
    for (name, reason) in &skipped {
        println!("skipped {name}: {reason}");
    }
}
