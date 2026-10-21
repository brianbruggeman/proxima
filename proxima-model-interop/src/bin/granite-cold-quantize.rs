use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::process;

use proxima_model_interop::cold_artifact::{
    ColdScaleMode, compare_checkpoint, convert_checkpoint_with_scale,
};
use proxima_primitives::float_layout::FloatLayout;

const REPOSITORY: &str = "ibm-granite/granite-4.0-micro-base";
const REVISION: &str = "98e1d641451818f6c8e6b17c1c359b5b4ac495d0";

fn main() {
    if let Err(error) = run() {
        eprintln!("granite-cold-quantize: {error}");
        process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let first_argument = arguments.next().ok_or_else(usage)?;
    if first_argument == "compare" {
        let source_dir = arguments.next().map(PathBuf::from).ok_or_else(usage)?;
        let artifact_dir = arguments.next().map(PathBuf::from).ok_or_else(usage)?;
        let report_path = arguments.next().map(PathBuf::from).ok_or_else(usage)?;
        if arguments.next().is_some() {
            return Err(usage());
        }
        let report =
            compare_checkpoint(&source_dir, &artifact_dir).map_err(|error| error.to_string())?;
        let report_file = File::create_new(&report_path)
            .map_err(|error| format!("cannot create report {report_path:?}: {error}"))?;
        serde_json::to_writer_pretty(&report_file, &report)
            .map_err(|error| format!("cannot write comparison report: {error}"))?;
        println!(
            "format={} tensors={} values={} changed_values={} saturated_values={} zeroed_nonzero_values={} max_abs_error={} report={}",
            report.format,
            report.tensors.len(),
            report.source_values,
            report.changed_values,
            report.saturated_values,
            report.zeroed_nonzero_values,
            report.maximum_absolute_error,
            report_path.display()
        );
        return Ok(());
    }
    let source_dir = PathBuf::from(first_argument);
    let output_dir = arguments.next().map(PathBuf::from).ok_or_else(usage)?;
    let layout_input = arguments.next().ok_or_else(usage)?;
    let chunk_elements = arguments
        .next()
        .ok_or_else(usage)?
        .parse::<usize>()
        .map_err(|error| format!("chunk element count is not an integer: {error}"))?;
    if arguments.next().is_some() {
        return Err(usage());
    }

    let (layout_input, scale_mode) = match layout_input.strip_suffix("-tensor-scaled") {
        Some(layout_input) => (layout_input, ColdScaleMode::TensorMaximum),
        None => (layout_input.as_str(), ColdScaleMode::None),
    };
    let layout = parse_layout(layout_input)?;
    let manifest = convert_checkpoint_with_scale(
        &source_dir,
        &output_dir,
        layout,
        REPOSITORY,
        REVISION,
        chunk_elements,
        scale_mode,
    )
    .map_err(|error| error.to_string())?;
    println!(
        "repository={} revision={}",
        manifest.repository, manifest.revision
    );
    println!(
        "format={} scale_mode={} tensors={}",
        manifest.format,
        manifest.scale_mode,
        manifest.tensors.len()
    );
    println!("source_bytes_processed={}", manifest.source_bytes_processed);
    println!("source_bytes_scanned={}", manifest.source_bytes_scanned);
    println!("output_bytes_written={}", manifest.output_bytes_written);
    println!("manifest={}", output_dir.join("manifest.json").display());
    Ok(())
}

fn parse_layout(input: &str) -> Result<FloatLayout, String> {
    match input {
        "e5m2" => Ok(FloatLayout::E5M2_SATURATING),
        "e3m2" => Ok(FloatLayout::E3M2),
        "e2m2" => Ok(FloatLayout::E2M2),
        _ => serde_json::from_str(input).map_err(|error| {
            format!("layout must be a known alias or FloatLayout object: {error}")
        }),
    }
}

fn usage() -> String {
    String::from(
        "usage: granite-cold-quantize <checkpoint-dir> <new-output-dir> <layout-alias|layout-json> <chunk-elements-multiple-of-8> | granite-cold-quantize compare <checkpoint-dir> <artifact-dir> <new-report.json>",
    )
}
