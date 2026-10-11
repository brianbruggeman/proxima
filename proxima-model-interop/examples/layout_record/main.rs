mod checkpoint;
mod codebook;
mod encoders;
mod error;
mod fixture;
mod incumbent;
mod linear;

use std::path::{Path, PathBuf};

use proxima_gguf::GgmlType;
use proxima_gguf::quant::dispatch;
use proxima_primitives::Codec;

use checkpoint::Checkpoint;
use error::RecordError;
use fixture::{Origin, Recorded};
use incumbent::{ELEMENTS, Incumbent, incumbent};

const CODEC_VARIANTS: usize = 33;

struct Sources {
    granite: Checkpoint,
    e2b: Checkpoint,
}

struct Arguments {
    granite: PathBuf,
    e2b: PathBuf,
    out: PathBuf,
}

fn flag_value(arguments: &[String], flag: &str) -> Result<PathBuf, RecordError> {
    arguments
        .iter()
        .position(|argument| argument == flag)
        .and_then(|index| arguments.get(index + 1))
        .map(PathBuf::from)
        .ok_or(RecordError::Usage)
}

fn parse_arguments(arguments: &[String]) -> Result<Arguments, RecordError> {
    Ok(Arguments {
        granite: flag_value(arguments, "--granite")?,
        e2b: flag_value(arguments, "--e2b")?,
        out: flag_value(arguments, "--out")?,
    })
}

fn checkpoint_choice(codec: Codec) -> Option<(bool, GgmlType, &'static str)> {
    match codec {
        Codec::Float16 => Some((false, GgmlType::F16, "")),
        Codec::Q4_0 => Some((false, GgmlType::Q4_0, "attn_k")),
        Codec::Q6K => Some((false, GgmlType::Q6_K, "")),
        Codec::Q8_0 => Some((true, GgmlType::Q8_0, "attn_q")),
        _ => None,
    }
}

fn granite_rows(sources: &Sources) -> Result<(Vec<f32>, String), RecordError> {
    let tensor = sources
        .granite
        .tensor_of(GgmlType::Q8_0, "attn_q", ELEMENTS)?;
    let rows = sources.granite.read_rows(tensor, ELEMENTS / 32 * 34)?;
    let mut values = vec![0.0_f32; ELEMENTS];
    dispatch::dequantize(GgmlType::Q8_0, &rows.bytes, &mut values)
        .map_err(|error| RecordError::quant("row dequantize", Codec::Q8_0, error))?;
    Ok((
        values,
        format!(
            "{} tensor {} byte_offset {} dequantized by the Q8_0 incumbent",
            sources.granite.blob_name(),
            rows.tensor,
            rows.byte_offset
        ),
    ))
}

fn from_checkpoint(
    codec: Codec,
    plan: Incumbent,
    sources: &Sources,
) -> Result<Option<(Vec<u8>, Origin)>, RecordError> {
    let Some((use_granite, kind, fragment)) = checkpoint_choice(codec) else {
        return Ok(None);
    };
    let checkpoint = if use_granite {
        &sources.granite
    } else {
        &sources.e2b
    };
    let tensor = checkpoint.tensor_of(kind, fragment, ELEMENTS)?;
    let rows = checkpoint.read_rows(tensor, plan.byte_len())?;
    let detail = format!(
        "{} tensor {} byte_offset {}",
        checkpoint.blob_name(),
        rows.tensor,
        rows.byte_offset
    );
    Ok(Some((rows.bytes, Origin::Checkpoint(detail))))
}

fn from_encoder(
    codec: Codec,
    plan: Incumbent,
    sources: &Sources,
) -> Result<(Vec<u8>, Origin), RecordError> {
    let (rows, detail) = granite_rows(sources)?;
    let blocks = encoders::encode(codec, plan, &rows)?;
    Ok((
        blocks,
        Origin::Encoder(format!("example encoder over {detail}")),
    ))
}

fn record(codec: Codec, sources: &Sources, out: &Path) -> Result<String, RecordError> {
    let Some(plan) = incumbent(codec) else {
        return fixture::write_unavailable(out, codec);
    };
    let (blocks, origin) = match from_checkpoint(codec, plan, sources)? {
        Some(found) => found,
        None => from_encoder(codec, plan, sources)?,
    };
    let mut decoded = vec![0.0_f32; ELEMENTS];
    plan.decode(codec, &blocks, &mut decoded)?;
    fixture::write(
        out,
        &Recorded {
            codec,
            origin,
            blocks,
            decoded,
        },
    )
}

fn run(arguments: &[String]) -> Result<(), RecordError> {
    let arguments = parse_arguments(arguments)?;
    let sources = Sources {
        granite: Checkpoint::open(&arguments.granite)?,
        e2b: Checkpoint::open(&arguments.e2b)?,
    };
    let mut recorded = 0;
    for tag in 0..=u8::MAX {
        let Some(codec) = Codec::from_tag(tag) else {
            continue;
        };
        println!("{}", record(codec, &sources, &arguments.out)?);
        recorded += 1;
    }
    if recorded == CODEC_VARIANTS {
        Ok(())
    } else {
        Err(RecordError::CodecCount {
            found: recorded,
            expected: CODEC_VARIANTS,
        })
    }
}

fn main() -> Result<(), RecordError> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    run(&arguments)
}
