//! `proxima model …` — model-file conversions. `export-mlx` writes an
//! mlx-lm model directory for a GGUF by calling
//! [`proxima_model_interop::export_mlx_directory`], which composes the
//! interop crate's plain or affine tensor export, config writer and
//! tokenizer writer; compose those directly to keep the bytes in memory.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Subcommand, ValueEnum};
use proxima_model_interop::{AffineParams, export_mlx_directory};
use proxima_telemetry::error;
use proxima_tensor::DType;

#[derive(Args, Debug)]
pub struct ModelArgs {
    #[command(subcommand)]
    action: ModelCommand,
}

#[derive(Subcommand, Debug)]
pub enum ModelCommand {
    /// Write an mlx-lm model directory (weights, config.json, tokenizer
    /// files) for a GGUF. Prints each file written.
    ExportMlx {
        /// Path to the source GGUF, e.g. an Ollama blob.
        model: PathBuf,

        /// Directory to write; created if absent, files overwritten.
        #[arg(long)]
        out: PathBuf,

        /// Affine-quantize every matrix as `BITS,GROUP_SIZE` (bits 2, 3, 4,
        /// 5, 6 or 8; group size 32, 64 or 128), e.g. `4,64`, mlx-lm's
        /// default. Omit for a plain float export.
        #[arg(long, value_name = "BITS,GROUP_SIZE", value_parser = parse_quantize)]
        quantize: Option<AffineParams>,

        /// Float dtype of plain tensors, scales and biases.
        #[arg(long, value_enum, default_value_t = FloatDtype::Bf16)]
        dtype: FloatDtype,
    },
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatDtype {
    Bf16,
    F16,
    F32,
}

impl FloatDtype {
    const fn dtype(self) -> DType {
        match self {
            Self::Bf16 => DType::BFloat16,
            Self::F16 => DType::Float16,
            Self::F32 => DType::Float32,
        }
    }
}

fn parse_quantize(raw: &str) -> Result<AffineParams, String> {
    let (bits, group_size) = raw
        .split_once(',')
        .ok_or_else(|| format!("expected 'BITS,GROUP_SIZE', got '{raw}'"))?;
    let bits = bits
        .trim()
        .parse::<u8>()
        .map_err(|error| format!("bits '{bits}': {error}"))?;
    let group_size = group_size
        .trim()
        .parse::<usize>()
        .map_err(|error| format!("group size '{group_size}': {error}"))?;
    AffineParams::new(bits, group_size).map_err(|error| error.to_string())
}

pub fn run(args: ModelArgs) -> ExitCode {
    match args.action {
        ModelCommand::ExportMlx {
            model,
            out,
            quantize,
            dtype,
        } => match export_mlx_directory(&model, &out, quantize, dtype.dtype()) {
            Ok(paths) => {
                paths.iter().for_each(|path| println!("{}", path.display()));
                ExitCode::SUCCESS
            }
            Err(error) => {
                error!(?error, model = %model.display(), "mlx export failed");
                ExitCode::from(1)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_four_bits_group_sixty_four_is_the_mlx_lm_default() {
        assert_eq!(parse_quantize("4,64"), Ok(AffineParams::default()));
    }

    #[test]
    fn quantize_accepts_the_corner_of_the_grid() {
        assert_eq!(
            parse_quantize("8, 128"),
            AffineParams::new(8, 128).map_err(|error| error.to_string())
        );
    }

    #[test]
    fn quantize_refuses_seven_bits_naming_the_bits() {
        let refusal = parse_quantize("7,64").err().unwrap_or_default();
        assert!(refusal.contains('7'), "{refusal}");
    }

    #[test]
    fn quantize_refuses_group_size_forty_eight() {
        let refusal = parse_quantize("4,48").err().unwrap_or_default();
        assert!(refusal.contains("48"), "{refusal}");
    }

    #[test]
    fn quantize_refuses_a_value_without_a_comma() {
        let refusal = parse_quantize("4").err().unwrap_or_default();
        assert!(refusal.contains("BITS,GROUP_SIZE"), "{refusal}");
    }

    #[test]
    fn quantize_refuses_a_non_numeric_bits_field() {
        let refusal = parse_quantize("four,64").err().unwrap_or_default();
        assert!(refusal.contains("four"), "{refusal}");
    }

    #[test]
    fn dtype_names_map_to_the_export_dtypes() {
        let mapped = [FloatDtype::Bf16, FloatDtype::F16, FloatDtype::F32].map(FloatDtype::dtype);
        assert_eq!(mapped, [DType::BFloat16, DType::Float16, DType::Float32]);
    }
}
