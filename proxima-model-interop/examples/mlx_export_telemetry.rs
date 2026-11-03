//! `mlx_export_telemetry <gguf> <out-dir> <events-file> [bits,group]` runs
//! `export_mlx_directory` with a file-sink recorder installed, so the
//! per-phase events carrying the bytes held at each phase land in
//! `<events-file>`.

use std::error::Error;
use std::path::PathBuf;

use proxima_model_interop::{AffineParams, export_mlx_directory};
use proxima_telemetry::emit::{EnvFilter, global};
use proxima_telemetry::export::{Exporter, Formatter};
use proxima_telemetry::recorder::Recorder;
use proxima_tensor::DType;

fn quantization(text: &str) -> Result<AffineParams, Box<dyn Error>> {
    let (bits, group) = text.split_once(',').ok_or("expected bits,group")?;
    Ok(AffineParams::new(bits.parse()?, group.parse()?)?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(gguf), Some(out_dir), Some(events)) = (
        args.next().map(PathBuf::from),
        args.next().map(PathBuf::from),
        args.next(),
    ) else {
        return Err(
            "usage: mlx_export_telemetry <gguf> <out-dir> <events-file> [bits,group]".into(),
        );
    };
    let quantize = args.next().map(|text| quantization(&text)).transpose()?;
    global::install(EnvFilter::parse("proxima_model_interop=debug"));
    let recorder = Recorder::builder()
        .export(Exporter::file(&events).format(Formatter::Text))?
        .install()?;
    let written = export_mlx_directory(&gguf, &out_dir, quantize, DType::BFLOAT16)?;
    while recorder.drain() > 0 {}
    println!("wrote {} files", written.len());
    Ok(())
}
