use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapOptions};
use proxima_gguf::{GgufError, parse_complete};
use proxima_model_interop::{
    GPU_LAYERS_ALL, InteropError, LoadedModel, ServingConfig, SpeculativeConfig,
    SpeculativeDecodeStats, SpeculativeType, SpeculativeTypeSet, TokenEvent,
};
use thiserror::Error;

const DRAFTER_TYPES: [SpeculativeType; 5] = [
    SpeculativeType::NgramSimple,
    SpeculativeType::NgramMapK,
    SpeculativeType::NgramMapK4v,
    SpeculativeType::NgramMod,
    SpeculativeType::NgramCache,
];
const MAX_TOKENS: usize = 64;

#[derive(Debug, Error)]
enum CorpusError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Interop(#[from] InteropError),
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("{0}")]
    InvalidInput(String),
}

struct CorpusPrompt {
    id: String,
    prompt: String,
}

fn read_corpus(path: &PathBuf) -> Result<Vec<CorpusPrompt>, CorpusError> {
    let file = File::open(path)?;
    let mut prompts = Vec::new();
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: serde_json::Value = serde_json::from_str(&line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{}: {error}", path.display(), line_number + 1),
            )
        })?;
        let id = record
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| invalid_record(path, line_number + 1, "id must be a string"))?;
        let prompt = record
            .get("prompt")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| invalid_record(path, line_number + 1, "prompt must be a string"))?;
        prompts.push(CorpusPrompt {
            id: id.to_string(),
            prompt: prompt.to_string(),
        });
    }
    Ok(prompts)
}

fn invalid_record(path: &Path, line_number: usize, detail: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{}:{line_number}: {detail}", path.display()),
    )
}

fn discard_token_event(_: TokenEvent<'_>) -> core::ops::ControlFlow<(), ()> {
    core::ops::ControlFlow::Continue(())
}

fn parse_usize(name: &str, value: &str) -> Result<usize, CorpusError> {
    value
        .parse::<usize>()
        .map_err(|error| CorpusError::InvalidInput(format!("{name} must be an integer: {error}")))
}

fn parse_i32(name: &str, value: &str) -> Result<i32, CorpusError> {
    value
        .parse::<i32>()
        .map_err(|error| CorpusError::InvalidInput(format!("{name} must be an integer: {error}")))
}

fn run() -> Result<(), CorpusError> {
    let args: Vec<String> = env::args().skip(1).collect();
    let (model_path, corpus_path, mut index) = if args.first().is_some_and(|arg| arg == "--corpus")
    {
        if args.len() < 3 {
            return Err(CorpusError::InvalidInput(
                "usage: speculative_acceptance_corpus --corpus CORPUS MODEL [OPTIONS]".into(),
            ));
        }
        (PathBuf::from(&args[2]), PathBuf::from(&args[1]), 3)
    } else {
        if args.len() < 2 {
            return Err(CorpusError::InvalidInput(
                "usage: speculative_acceptance_corpus MODEL CORPUS [OPTIONS]".into(),
            ));
        }
        (PathBuf::from(&args[0]), PathBuf::from(&args[1]), 2)
    };
    let mut max_tokens = MAX_TOKENS;
    let mut prompt_limit = usize::MAX;
    let mut gpu_layers = 0;
    let mut prompt_id = None;
    while index < args.len() {
        let flag = &args[index];
        let value = args.get(index + 1).ok_or_else(|| {
            CorpusError::InvalidInput(format!("option {flag:?} requires a value"))
        })?;
        match flag.as_str() {
            "--max-tokens" => max_tokens = parse_usize("max tokens", value)?,
            "--limit-prompts" => prompt_limit = parse_usize("prompt limit", value)?,
            "--prompt-id" => prompt_id = Some(value.clone()),
            "--gpu-layers" => {
                gpu_layers = if value == "all" {
                    GPU_LAYERS_ALL
                } else {
                    parse_i32("GPU layers", value)?
                };
            }
            _ => {
                return Err(CorpusError::InvalidInput(format!(
                    "unknown option {flag:?}"
                )));
            }
        }
        index += 2;
    }

    let prompts: Vec<CorpusPrompt> = read_corpus(&corpus_path)?
        .into_iter()
        .filter(|record| prompt_id.as_deref().is_none_or(|id| record.id == id))
        .take(prompt_limit)
        .collect();
    if prompts.is_empty() {
        return Err(CorpusError::InvalidInput(
            "the corpus selection contains no prompt records".into(),
        ));
    }

    let model_file = File::open(&model_path)?;
    // SAFETY: `model_file` remains alive while `model_bytes` is borrowed.
    let model_bytes: Mmap = unsafe { MmapOptions::new().map(&model_file)? };
    let parsed = parse_complete(&model_bytes)?;
    let model = LoadedModel::load(&parsed, &model_bytes)?;

    println!("environment_os = {}", env::consts::OS);
    println!("environment_arch = {}", env::consts::ARCH);
    println!("gpu_layers = {gpu_layers}");
    println!("model = {}", model_path.display());
    println!("corpus = {}", corpus_path.display());
    println!("max_tokens = {max_tokens}");
    println!("prompts = {}", prompts.len());
    let mut successful_prompts = 0usize;
    let mut failed_prompts = 0usize;
    for drafter_type in DRAFTER_TYPES {
        let serving_config = ServingConfig {
            gpu_layers,
            speculative: SpeculativeConfig {
                speculative_types: SpeculativeTypeSet::single(drafter_type),
                ..SpeculativeConfig::none()
            },
            ..ServingConfig::default()
        };
        let mut verify_steps = 0u64;
        let mut drafted_tokens = 0u64;
        let mut accepted_tokens = 0u64;

        for corpus_prompt in &prompts {
            let mut stats = SpeculativeDecodeStats::default();
            let mut on_token = discard_token_event;
            let result = model.generate_streaming_with_speculative_stats(
                &corpus_prompt.prompt,
                max_tokens,
                serving_config,
                &mut on_token,
                &mut stats,
                None,
            );
            let generated_ids = match result {
                Ok((generated_ids, _, _)) => generated_ids,
                Err(error) => {
                    failed_prompts += 1;
                    eprintln!(
                        "prompt_error drafter={} id={} error={error}",
                        drafter_type.llama_name(),
                        corpus_prompt.id
                    );
                    continue;
                }
            };

            successful_prompts += 1;
            let type_stats = stats.per_type_stats(drafter_type);
            println!(
                "prompt drafter={} id={} ids={generated_ids:?} verify_steps={} drafted={} accepted={}",
                drafter_type.llama_name(),
                corpus_prompt.id,
                stats.verify_steps,
                type_stats.drafted,
                type_stats.accepted
            );
            verify_steps += stats.verify_steps;
            drafted_tokens += type_stats.drafted;
            accepted_tokens += type_stats.accepted;
        }

        let mean_accepted_per_step = if verify_steps == 0 {
            0.0
        } else {
            accepted_tokens as f64 / verify_steps as f64
        };
        println!(
            "summary drafter={} verify_steps={} drafted={} accepted={} mean_accepted_per_step={mean_accepted_per_step:.6}",
            drafter_type.llama_name(),
            verify_steps,
            drafted_tokens,
            accepted_tokens
        );
    }
    println!("successful_prompts = {successful_prompts}");
    println!("failed_prompts = {failed_prompts}");
    if failed_prompts > 0 {
        return Err(CorpusError::InvalidInput(format!(
            "{failed_prompts} prompt generation attempts failed"
        )));
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
