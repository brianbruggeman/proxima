//! Converts checkpoints offline:
//! `convert_codec --codec fp6_e3m2 --policy default --job <source.gguf>=<output.gguf> ...`.
//!
//! Composes `parse_complete` (the source directory), `convert_codec` (re-encode the
//! tensors the policy names, carry the rest byte for byte) and one file write per
//! job. `--codec` takes the `[codec_policy]` names (`shipped`, `fp6_e3m2`);
//! `--policy default` keeps the default tensor classes, any other value is a
//! comma-separated class list such as `attn_q,attn_k`. Each job prints one line:
//! `converted: <n> tensors, kept: <n>, bytes: <source> -> <output>`.

use std::error::Error;
use std::fs::{File, create_dir_all, write};
use std::path::PathBuf;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_model_interop::{CodecPolicySettings, CodecPolicyWeights, convert_codec};

const USAGE: &str = "usage: convert_codec --codec <name> --policy <default|class,class> --job <source>=<output> ...";

struct Job {
    source: PathBuf,
    output: PathBuf,
}

fn parse_job(raw: &str) -> Result<Job, Box<dyn Error>> {
    let (source, output) = raw
        .split_once('=')
        .ok_or_else(|| format!("job {raw:?} is not <source>=<output>"))?;
    if source == output {
        return Err(format!("job {raw:?} would overwrite its own source").into());
    }
    Ok(Job {
        source: PathBuf::from(source),
        output: PathBuf::from(output),
    })
}

fn parse_weights(name: &str) -> Result<CodecPolicyWeights, Box<dyn Error>> {
    serde_json::from_value(serde_json::Value::String(name.to_owned()))
        .map_err(|error| format!("unknown codec {name:?}: {error}").into())
}

fn parse_policy(
    weights: CodecPolicyWeights,
    policy: &str,
) -> Result<CodecPolicySettings, Box<dyn Error>> {
    let builder = CodecPolicySettings::builder().weights(weights);
    match policy {
        "default" => Ok(builder.build()),
        "" => Err("policy is empty: default or a class list".into()),
        classes => Ok(builder
            .classes(classes.split(',').map(str::to_owned).collect())
            .build()),
    }
}

#[derive(Default)]
struct Flags {
    codec: Option<String>,
    policy: Option<String>,
    jobs: Vec<Job>,
}

fn read_flags(mut arguments: impl Iterator<Item = String>) -> Result<Flags, Box<dyn Error>> {
    let mut flags = Flags::default();
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("{flag} has no value"))?;
        match flag.as_str() {
            "--codec" => flags.codec = Some(value),
            "--policy" => flags.policy = Some(value),
            "--job" => flags.jobs.push(parse_job(&value)?),
            other => return Err(format!("unknown flag {other}\n{USAGE}").into()),
        }
    }
    Ok(flags)
}

fn parse_args() -> Result<(CodecPolicySettings, Vec<Job>), Box<dyn Error>> {
    let flags = read_flags(std::env::args().skip(1))?;
    let (Some(codec), Some(policy)) = (flags.codec, flags.policy) else {
        return Err(format!("--codec and --policy are required\n{USAGE}").into());
    };
    if flags.jobs.is_empty() {
        return Err(format!("no --job given\n{USAGE}").into());
    }
    Ok((parse_policy(parse_weights(&codec)?, &policy)?, flags.jobs))
}

fn run_job(job: &Job, policy: &CodecPolicySettings) -> Result<String, Box<dyn Error>> {
    let file = File::open(&job.source)?;
    // the checkpoint is not written while this process reads it
    let mapping = unsafe { Mmap::map(&file)? };
    let parsed = parse_complete(&mapping)?;
    let (converted_file, converted) = convert_codec(&parsed, &mapping, policy)?;
    if let Some(directory) = job.output.parent() {
        create_dir_all(directory)?;
    }
    write(&job.output, &converted_file)?;
    Ok(format!(
        "converted: {converted} tensors, kept: {}, bytes: {} -> {}",
        parsed.tensors.len() - converted,
        mapping.len(),
        converted_file.len()
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let (policy, jobs) = parse_args()?;
    for job in &jobs {
        let summary =
            run_job(job, &policy).map_err(|error| format!("{}: {error}", job.source.display()))?;
        println!("{summary}");
    }
    Ok(())
}
